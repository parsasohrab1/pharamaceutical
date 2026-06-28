#!/usr/bin/env python3
"""Run HQCA API server."""

import uvicorn

if __name__ == "__main__":
    import os
    port = int(os.getenv("HQCA_PORT", "18080"))
    uvicorn.run("api:app", host="127.0.0.1", port=port, reload=True)
