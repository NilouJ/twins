# this app is API wrapping fast api for document pdf analysis with textract and claude bedrock

import os
import re
import json
import uuid
import time
import asyncio
from datetime import datetime, timezone
from typing import Any, Dict, Tuple
from urllib.parse import urlparse

import boto3
from fastapi import FastAPI, HTTPException
from mangum import Mangum
from anthropic import AsyncAnthropicBedrock


AWS_REGION = os.getenv("AWS_REGION", "ap-southeast-2")
MODEL_ID = os.getenv("BEDROCK_MODEL_ID", "anthropic.claude-3-haiku-20240307-v1:0")
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "1500"))
TEMPERATURE = float(os.getenv("TEMPERATURE", "0"))
CONCURRENCY = int(os.getenv("CONCURRENCY", "3"))

s3 = boto3.client("s3", region_name=AWS_REGION)
textract = boto3.client("textract", region_name=AWS_REGION)
claude = AsyncAnthropicBedrock(aws_region=AWS_REGION)
sem = asyncio.Semaphore(CONCURRENCY)

app = FastAPI(title="Twins Document Classifier")


def parse_s3_uri(uri: str) -> Tuple[str, str]:
    parsed = urlparse(uri)
    if parsed.scheme != "s3" or not parsed.netloc or not parsed.path:
        raise ValueError(f"Invalid S3 URI: {uri}")
    return parsed.netloc, parsed.path.lstrip("/")


def make_s3_uri(bucket: str, key: str) -> str:
    return f"s3://{bucket}/{key}"


def doc_id_from_key(key: str) -> str:
    filename = key.rsplit("/", 1)[-1]
    name = filename.rsplit(".", 1)[0]
    name = re.sub(r"[^A-Za-z0-9._-]+", "-", name).strip("-")
    return name or f"document-{uuid.uuid4().hex[:8]}"


def normalise_prefix(prefix_s3_uri: str) -> Tuple[str, str]:
    bucket, prefix = parse_s3_uri(prefix_s3_uri)
    prefix = prefix.strip("/")
    if prefix:
        prefix += "/"
    return bucket, prefix


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def write_json(payload: Dict[str, Any], output_s3_uri: str) -> str:
    bucket, key = parse_s3_uri(output_s3_uri)
    s3.put_object(
        Bucket=bucket,
        Key=key,
        Body=json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8"),
        ContentType="application/json",
    )
    return output_s3_uri


async def extract_text(source: str) -> str:
    def _detect():
        bucket, key = parse_s3_uri(source)

        # Start async Textract job
        start = textract.start_document_text_detection(
            DocumentLocation={"S3Object": {"Bucket": bucket, "Name": key}}
        )
        job_id = start["JobId"]

        # Poll until complete
        while True:
            resp = textract.get_document_text_detection(JobId=job_id)
            status = resp["JobStatus"]
            if status in ("SUCCEEDED", "FAILED"):
                break
            time.sleep(3)

        if status == "FAILED":
            raise RuntimeError(f"Textract job failed: {job_id}")

        # Collect all paginated blocks
        blocks = resp["Blocks"]
        next_token = resp.get("NextToken")
        while next_token:
            page = textract.get_document_text_detection(
                JobId=job_id, NextToken=next_token
            )
            blocks.extend(page["Blocks"])
            next_token = page.get("NextToken")

        # Join all LINE blocks into text
        lines = [b["Text"] for b in blocks if b["BlockType"] == "LINE"]
        return "\n".join(lines)

    return await asyncio.to_thread(_detect)


async def extract_text_start_s3(source: str) -> str:
    bucket, key = parse_s3_uri(source)

    # Start async Textract job
    start = textract.start_document_text_detection(
        DocumentLocation={"S3Object": {"Bucket": bucket, "Name": key}}
    )
    job_id = start["JobId"]

    # Saving started job_ids and status on s3
    resp = textract.get_document_text_detection(JobId=job_id)
    status = resp["JobStatus"]
    s3.put_object(
        Bucket=bucket,
        Key=f"textract_jobs/{job_id}.json",
        Body=json.dumps({"job_id": job_id, "status": status}).encode("utf-8"),
        ContentType="application/json",
    )
    return job_id


async def extract_text_collect_s3(job_id_json: str) -> str:
    bucket, key = parse_s3_uri(job_id_json)

    # read the json from s3 and parse it into a dict
    data = json.loads(s3.get_object(Bucket=bucket, Key=key)["Body"].read())
    job_id = data["job_id"]

    # check the job status
    resp = textract.get_document_text_detection(JobId=job_id)
    status = resp["JobStatus"]

    if status == "FAILED":
        raise RuntimeError(f"Textract job failed: {job_id}")

    if status != "SUCCEEDED":
        return None

    # Collect all paginated blocks
    blocks = resp["Blocks"]
    next_token = resp.get("NextToken")
    while next_token:
        page = textract.get_document_text_detection(
            JobId=job_id, NextToken=next_token
        )
        blocks.extend(page["Blocks"])
        next_token = page.get("NextToken")

    # Join all LINE blocks into text
    lines = [b["Text"] for b in blocks if b["BlockType"] == "LINE"]

    # save results in S3
    s3.put_object(
        Bucket=bucket,
        Key=f"textract_jobs_results/{job_id}.json",
        Body=json.dumps({"job_id": job_id, "lines": lines}).encode("utf-8"),
        ContentType="application/json",
    )

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


async def classify_with_claude(doc_id: str, extracted_text: str) -> Dict[str, Any]:
    prompt = build_prompt(doc_id, extracted_text)

    async with sem:
        response = await claude.messages.create(
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


@app.get("/")
def root():
    return {
        "app": "twins-document-classifier",
        "status": "running",
    }

@app.post("/api/ocr/start")
async def start_ocr(payload: Dict[str, Any]):
    input_pdf_s3_uri = payload.get("input_pdf_s3_uri")
    if not input_pdf_s3_uri:
        raise HTTPException(status_code=400, detail="Missing input_pdf_s3_uri")

    try:
        job_id = await extract_text_start_s3(input_pdf_s3_uri)
        return {
            "status": "started",
            "job_id": job_id,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/ocr/collect")
async def collect_ocr(payload: Dict[str, Any]):
    job_id_json = payload.get("job_id_json")
    if not job_id_json:
        raise HTTPException(status_code=400, detail="Missing job_id_json")

    try:
        extracted_text = await extract_text_collect_s3(job_id_json)
        if extracted_text is None:
            return {
                "status": "pending",
                "message": "Textract job is still in progress. Please check again later.",
            }
        else:
            return {
                "status": "completed",
                "extracted_text": extracted_text,
            }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/classify")
async def classify(payload: Dict[str, Any]):
    doc_id = payload.get("doc_id")
    extracted_text = payload.get("extracted_text")

    if not doc_id:
        raise HTTPException(status_code=400, detail="Missing doc_id")

    if not extracted_text:
        raise HTTPException(status_code=400, detail="Missing extracted_text")

    try:
        classification = await classify_with_claude(doc_id, extracted_text)
        return {
            "status": "completed",
            "classification": classification,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/classify-document")
async def classify_document(payload: Dict[str, Any]):
    input_pdf_s3_uri = payload.get("input_pdf_s3_uri")
    output_prefix_s3_uri = payload.get("output_prefix_s3_uri")

    if not input_pdf_s3_uri:
        raise HTTPException(status_code=400, detail="Missing input_pdf_s3_uri")

    if not output_prefix_s3_uri:
        raise HTTPException(status_code=400, detail="Missing output_prefix_s3_uri")

    try:
        _, input_key = parse_s3_uri(input_pdf_s3_uri)
        output_bucket, output_prefix = normalise_prefix(output_prefix_s3_uri)

        doc_id = doc_id_from_key(input_key)

        raw_output_s3_uri = make_s3_uri(
            output_bucket,
            f"{output_prefix}{doc_id}_raw.json",
        )

        classification_output_s3_uri = make_s3_uri(
            output_bucket,
            f"{output_prefix}{doc_id}_classification.json",
        )

        extracted_text = await extract_text(input_pdf_s3_uri)

        raw_payload = {
            "doc_id": doc_id,
            "source_pdf_s3_uri": input_pdf_s3_uri,
            "generated_at": now_utc(),
            "raw": {
                "extracted_text": extracted_text,
            },
            "stats": {
                "text_chars": len(extracted_text),
            },
        }

        write_json(raw_payload, raw_output_s3_uri)

        classification = await classify_with_claude(doc_id, extracted_text)

        classification_payload = {
            "doc_id": doc_id,
            "source_pdf_s3_uri": input_pdf_s3_uri,
            "raw_output_s3_uri": raw_output_s3_uri,
            "generated_at": now_utc(),
            "model_id": MODEL_ID,
            "classification": classification,
        }

        write_json(classification_payload, classification_output_s3_uri)

        return {
            "status": "completed",
            "doc_id": doc_id,
            "input_pdf_s3_uri": input_pdf_s3_uri,
            "raw_output_s3_uri": raw_output_s3_uri,
            "classification_output_s3_uri": classification_output_s3_uri,
            "stats": {
                "text_chars": len(extracted_text),
            },
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


handler = Mangum(app)