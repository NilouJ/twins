import os
import re
import json
import uuid
from datetime import datetime, timezone
from typing import Any, Dict

import boto3
from anthropic import AnthropicBedrock


AWS_REGION = os.getenv("AWS_REGION", "ap-southeast-2")
TEXTRACT_RESULTS_BUCKET = os.getenv("TEXTRACT_RESULTS_BUCKET", "twins-lambdas")
MODEL_ID = os.getenv("BEDROCK_MODEL_ID", "anthropic.claude-3-haiku-20240307-v1:0")
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "1500"))
TEMPERATURE = float(os.getenv("TEMPERATURE", "0"))

s3 = boto3.client("s3", region_name=AWS_REGION)
textract = boto3.client("textract", region_name=AWS_REGION)
claude = AnthropicBedrock(aws_region=AWS_REGION)


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def doc_id_from_key(key: str) -> str:
    filename = key.rsplit("/", 1)[-1]
    name = filename.rsplit(".", 1)[0]
    name = re.sub(r"[^A-Za-z0-9._-]+", "-", name).strip("-")
    return name or f"document-{uuid.uuid4().hex[:8]}"


def collect_textract_text(job_id: str) -> str:
    response = textract.get_document_text_detection(JobId=job_id)

    status = response["JobStatus"]

    if status == "FAILED":
        raise RuntimeError(f"Textract job failed: {job_id}")

    if status != "SUCCEEDED":
        raise RuntimeError(f"Textract job not ready: {job_id}, status={status}")

    blocks = response.get("Blocks", [])
    next_token = response.get("NextToken")

    while next_token:
        page = textract.get_document_text_detection(
            JobId=job_id,
            NextToken=next_token,
        )
        blocks.extend(page.get("Blocks", []))
        next_token = page.get("NextToken")

    lines = [
        block["Text"]
        for block in blocks
        if block.get("BlockType") == "LINE" and "Text" in block
    ]

    return "\n".join(lines)


def build_prompt(doc_id: str, extracted_text: str) -> str:
    return f"""
You are classifying a document extracted with Amazon Textract.

Document ID:
{doc_id}

Extracted text:
{extracted_text}

Return ONLY valid JSON with this structure:

{{
  "document_classification": {{
    "document_type": "",
    "confidence": "low|medium|high",
    "reason": ""
  }},
  "page_classification": [
    {{
      "page": 1,
      "page_type": "",
      "confidence": "low|medium|high",
      "reason": ""
    }}
  ],
  "important_fields": {{
    "client_name": "",
    "document_date": "",
    "effective_date": "",
    "fees_or_charges_present": true,
    "tables_present": true
  }}
}}

Rules:
- Do not invent facts.
- Use empty strings if unknown.
- If page boundaries are unclear, return one page item with page = 1.
- JSON only. No markdown.
"""


def classify_with_claude(doc_id: str, extracted_text: str) -> Dict[str, Any]:
    prompt = build_prompt(doc_id, extracted_text)

    response = claude.messages.create(
        model=MODEL_ID,
        max_tokens=MAX_TOKENS,
        temperature=TEMPERATURE,
        messages=[{"role": "user", "content": prompt}],
    )

    text = "".join(
        block.text for block in response.content
        if hasattr(block, "text")
    ).strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {
            "document_classification": {
                "document_type": "unknown",
                "confidence": "low",
                "reason": "Model did not return valid JSON.",
            },
            "page_classification": [],
            "important_fields": {},
            "raw_model_response": text,
        }


def write_result(
    job_id: str,
    doc_id: str,
    extracted_text: str,
    classification: Dict[str, Any],
    sns_message: Dict[str, Any],
) -> str:
    key = f"textract_jobs_results/{job_id}.json"

    payload = {
        "job_id": job_id,
        "doc_id": doc_id,
        "status": "completed",
        "extracted_text": extracted_text,
        "lines": extracted_text.splitlines(),
        "classification": classification,
        "completed_at": now_utc(),
        "source": "sns",
        "sns_message": sns_message,
    }

    s3.put_object(
        Bucket=TEXTRACT_RESULTS_BUCKET,
        Key=key,
        Body=json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8"),
        ContentType="application/json",
    )

    return f"s3://{TEXTRACT_RESULTS_BUCKET}/{key}"


def handler(event, context):
    processed = []

    for record in event.get("Records", []):
        raw_message = record.get("Sns", {}).get("Message", "{}")

        try:
            message = json.loads(raw_message)
        except json.JSONDecodeError:
            processed.append({
                "status": "error",
                "reason": "Invalid SNS message JSON",
                "raw_message": raw_message,
            })
            continue

        job_id = message.get("JobId")
        textract_status = message.get("Status")

        if not job_id:
            processed.append({
                "status": "error",
                "reason": "Missing JobId",
                "message": message,
            })
            continue

        if textract_status != "SUCCEEDED":
            processed.append({
                "job_id": job_id,
                "status": textract_status,
                "reason": "Textract job did not succeed",
            })
            continue

        try:
            source_key = message.get("DocumentLocation", {}).get("S3ObjectName", "")
            doc_id = doc_id_from_key(source_key) if source_key else job_id

            extracted_text = collect_textract_text(job_id)
            classification = classify_with_claude(doc_id, extracted_text)
            output_s3_uri = write_result(job_id, doc_id, extracted_text, classification, message)

            processed.append({
                "job_id": job_id,
                "doc_id": doc_id,
                "status": "completed",
                "output_s3_uri": output_s3_uri,
                "text_chars": len(extracted_text),
            })

        except Exception as error:
            processed.append({
                "job_id": job_id,
                "status": "error",
                "error": str(error),
            })

    return {
        "statusCode": 200,
        "body": json.dumps({
            "processed": processed,
        }),
    }