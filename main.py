"""Entry point: start the FastAPI backend server."""

import logging
import os


def main():
    import uvicorn

    from dotenv import load_dotenv

    load_dotenv()

    log_level = os.getenv("LOG_LEVEL", "INFO").upper()
    logging.basicConfig(
        level=getattr(logging, log_level, logging.INFO),
        format="%(asctime)s %(levelname)-8s [%(name)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    logging.getLogger(__name__).info("Starting Segment2Explain FastAPI server")
    port = int(os.getenv("PORT", "8001"))
    uvicorn.run("backend.main:app", host="0.0.0.0", port=port, reload=False)


if __name__ == "__main__":
    main()
