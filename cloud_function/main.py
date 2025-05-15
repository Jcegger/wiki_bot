import logging
from flask import Request
import functions_framework
from core_script import run_pipeline

@functions_framework.http
def main(request: Request):
    try:
        logging.info("▶️ Starting run_pipeline from HTTP trigger")
        run_pipeline()
        return "✅ Script executed successfully", 200
    except Exception as e:
        logging.exception("❌ Pipeline crashed:")
        return f"❌ Error: {str(e)}", 500
