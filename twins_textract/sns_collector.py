import os
import json
from datetime import datetime, timezone
from typing import Any, Dict

import boto3


AWS_REGION = os.getenv("AWS_REGION", "ap-southeast-2")
TEXTRACT_RESULTS_BUCKET = os.getenv("TEXTRACT_RESULTS_BUCKET", "twins-lambdas")

s3 = boto3.client("s3", region_name=AWS_REGION)
textract = boto3.client("textract", region_name=AWS_REGION)


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


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


def write_ocr_result(job_id: str, extracted_text: str, sns_message: Dict[str, Any]) -> str:
    key = f"textract_jobs_results/{job_id}.json"

    payload = {
        "job_id": job_id,
        "status": "completed",
        "extracted_text": extracted_text,
        "lines": extracted_text.splitlines(),
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
            extracted_text = collect_textract_text(job_id)
            output_s3_uri = write_ocr_result(job_id, extracted_text, message)

            processed.append({
                "job_id": job_id,
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