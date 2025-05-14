from flask import Request
import functions_framework
from core_script import run_pipeline

@functions_framework.http
def main(request: Request):
    try:
        run_pipeline()
        return "✅ Script executed successfully", 200
    except Exception as e:
        return f"❌ Error: {str(e)}", 500