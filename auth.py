import os
from fastapi import Header, HTTPException, status

async def verify_token(authorization: str = Header(...)):
    """
    FastAPI dependency to verify the bearer token.
    """
    expected_token = os.getenv("API_BEARER_TOKEN")
    
    if not authorization.startswith("Bearer "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid Authorization header format. Use 'Bearer <token>'.",
        )
    
    token = authorization.split(" ")[1]
    
    if not token or token != expected_token:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid or expired token!",
        )