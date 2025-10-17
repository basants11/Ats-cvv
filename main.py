"""
AI Fusion Core - API Gateway Service
Main application entry point for the microservices-based AI platform
"""

import time
import uvicorn
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Optional, Dict, Any
import asyncio

from fastapi import FastAPI, Request, HTTPException, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import structlog
import grpc
import httpx

from app.config.settings import settings
from app.config.logging import configure_logging

# Configure structured logging
configure_logging()
logger = structlog.get_logger()

# Global state
startup_time: Optional[datetime] = None
grpc_channels: Dict[str, grpc.aio.Channel] = {}
http_clients: Dict[str, httpx.AsyncClient] = {}

# Service discovery configuration
SERVICES = {
    "ai-kernel": {"host": "localhost", "grpc_port": 50051, "http_port": 8001},
    "identity": {"host": "localhost", "grpc_port": 50052, "http_port": 8002},
    "cv-engine": {"host": "localhost", "grpc_port": 50053, "http_port": 8003},
    "conversational": {"host": "localhost", "grpc_port": 50054, "http_port": 8004},
    "analytics": {"host": "localhost", "grpc_port": 50055, "http_port": 8005},
    "automation": {"host": "localhost", "grpc_port": 50056, "http_port": 8006},
    "vision": {"host": "localhost", "grpc_port": 50057, "http_port": 8007},
    "plugin": {"host": "localhost", "grpc_port": 50058, "http_port": 8008},
}

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handle application startup and shutdown events"""
    global startup_time, grpc_channels, http_clients

    # Startup
    logger.info("Starting AI Fusion Core API Gateway", version=settings.VERSION)
    startup_time = datetime.utcnow()

    # Initialize service connections
    await initialize_service_connections()
    logger.info("Service connections initialized")

    # Health check all services
    await health_check_services()
    logger.info("All services healthy")

    yield

    # Shutdown
    logger.info("Shutting down AI Fusion Core API Gateway")

    # Close gRPC channels
    for service_name, channel in grpc_channels.items():
        logger.info(f"Closing gRPC channel for {service_name}")
        await channel.close()

    # Close HTTP clients
    for service_name, client in http_clients.items():
        logger.info(f"Closing HTTP client for {service_name}")
        await client.aclose()

async def initialize_service_connections():
    """Initialize connections to all microservices"""
    global grpc_channels, http_clients

    for service_name, config in SERVICES.items():
        try:
            # Initialize gRPC channel
            channel = grpc.aio.insecure_channel(
                f"{config['host']}:{config['grpc_port']}"
            )
            grpc_channels[service_name] = channel

            # Initialize HTTP client for REST fallback
            client = httpx.AsyncClient(
                base_url=f"http://{config['host']}:{config['http_port']}",
                timeout=30.0
            )
            http_clients[service_name] = client

            logger.info(f"Initialized connection to {service_name}")

        except Exception as e:
            logger.error(f"Failed to connect to {service_name}", error=str(e))

async def health_check_services() -> bool:
    """Health check all microservices"""
    # This would implement actual health checks to each service
    # For now, just check if channels are ready
    for service_name, channel in grpc_channels.items():
        try:
            await channel.channel_ready()
            logger.info(f"Service {service_name} is healthy")
        except Exception as e:
            logger.error(f"Service {service_name} health check failed", error=str(e))
            raise

# Create FastAPI application
app = FastAPI(
    title="AI Fusion Core API Gateway",
    description="Microservices-based AI platform for ATS-friendly CV building and advanced AI capabilities",
    version=settings.VERSION,
    openapi_url=f"{settings.API_V1_STR}/openapi.json",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Mount static files and templates for backward compatibility
app.mount("/static", StaticFiles(directory="app/static"), name="static")
templates = Jinja2Templates(directory="app/templates")

# Set up CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.BACKEND_CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Set up trusted host middleware for security
app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=settings.ALLOWED_HOSTS
)

# Enhanced middleware for request logging and timing
@app.middleware("http")
async def logging_middleware(request: Request, call_next):
    """Log all requests with timing information"""
    start_time = time.time()

    # Log request
    logger.info(
        "Gateway request started",
        method=request.method,
        path=request.url.path,
        client=request.client.host if request.client else "unknown",
        user_agent=request.headers.get("user-agent", "unknown")
    )

    try:
        response = await call_next(request)

        # Calculate processing time
        process_time = time.time() - start_time

        # Log response
        logger.info(
            "Gateway request completed",
            method=request.method,
            path=request.url.path,
            status_code=response.status_code,
            process_time=f"{process_time:.3f}s"
        )

        # Add timing header
        response.headers["X-Process-Time"] = str(process_time)
        response.headers["X-Service-Name"] = "api-gateway"

        return response

    except Exception as e:
        process_time = time.time() - start_time
        logger.error(
            "Gateway request failed",
            method=request.method,
            path=request.url.path,
            error=str(e),
            process_time=f"{process_time:.3f}s"
        )
        raise

# Security headers middleware
@app.middleware("http")
async def add_security_headers(request: Request, call_next):
    """Add security headers to all responses"""
    response = await call_next(request)

    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    response.headers["Content-Security-Policy"] = "default-src 'self'"

    return response

# Service routing middleware
@app.middleware("http")
async def service_routing_middleware(request: Request, call_next):
    """Route requests to appropriate microservices"""
    # Define service routing rules
    path = request.url.path

    # Route to specific services based on path
    if path.startswith("/api/v1/ai/") or path.startswith("/api/v2/ai/"):
        # Route to AI Kernel service
        return await route_to_service(request, call_next, "ai-kernel")
    elif path.startswith("/api/v1/auth/") or path.startswith("/api/v1/users/"):
        # Route to Identity service
        return await route_to_service(request, call_next, "identity")
    elif path.startswith("/api/v1/cv/") or path.startswith("/api/v2/cv/"):
        # Route to CV Engine service
        return await route_to_service(request, call_next, "cv-engine")
    elif path.startswith("/api/v1/analytics/") or path.startswith("/api/v2/analytics/"):
        # Route to Analytics service
        return await route_to_service(request, call_next, "analytics")
    else:
        # Handle locally or return 404
        return await call_next(request)

async def route_to_service(request: Request, call_next, service_name: str):
    """Route request to specific microservice"""
    # For now, pass through to existing routes
    # In full implementation, this would proxy to microservices
    return await call_next(request)

# Import routes after middleware setup to avoid circular imports
from app.routes import router

# Include API routes (backward compatibility)
app.include_router(router, prefix=settings.API_V1_STR)

# Gateway-specific endpoints
@app.get("/health")
async def gateway_health_check():
    """Enhanced health check endpoint for API Gateway"""
    global startup_time
    uptime = "unknown"
    if startup_time:
        uptime = str(datetime.utcnow() - startup_time)

    # Check service health
    service_status = {}
    for service_name in SERVICES.keys():
        try:
            if service_name in grpc_channels:
                await grpc_channels[service_name].channel_ready()
                service_status[service_name] = "healthy"
            else:
                service_status[service_name] = "disconnected"
        except Exception as e:
            service_status[service_name] = f"error: {str(e)}"

    return {
        "status": "healthy",
        "service": "api-gateway",
        "version": settings.VERSION,
        "uptime": uptime,
        "timestamp": datetime.utcnow().isoformat(),
        "services": service_status
    }

@app.get("/")
async def root():
    """Root endpoint with AI Fusion Core branding"""
    return {
        "message": "AI Fusion Core API Gateway",
        "description": "Microservices-based AI platform for ATS-friendly CV building and advanced AI capabilities",
        "docs": "/docs",
        "version": settings.VERSION,
        "health": "/health",
        "services": list(SERVICES.keys())
    }

@app.get("/api/v1/info")
async def api_info():
    """Get comprehensive API information"""
    global startup_time
    uptime = "unknown"
    if startup_time:
        uptime = str(datetime.utcnow() - startup_time)

    return {
        "name": "AI Fusion Core",
        "version": settings.VERSION,
        "description": "Microservices-based AI platform",
        "uptime": uptime,
        "start_time": startup_time.isoformat() if startup_time else None,
        "endpoints": {
            "docs": "/docs",
            "redoc": "/redoc",
            "openapi": f"{settings.API_V1_STR}/openapi.json",
            "health": "/health",
            "info": "/api/v1/info"
        },
        "microservices": {
            name: {
                "grpc_port": config["grpc_port"],
                "http_port": config["http_port"],
                "status": "configured"
            }
            for name, config in SERVICES.items()
        },
        "features": [
            "AI-powered CV analysis",
            "ATS-friendly CV generation",
            "Multiple CV templates",
            "PDF export",
            "Cloud storage integration",
            "Multi-agent AI orchestration",
            "Vector memory storage",
            "Real-time AI inference",
            "Plugin ecosystem",
            "Computer vision AI"
        ]
    }

# New v2 API endpoints for AI Fusion Core
@app.get("/api/v2/info")
async def api_v2_info():
    """Get AI Fusion Core v2 API information"""
    return {
        "name": "AI Fusion Core v2",
        "version": settings.VERSION,
        "architecture": "Microservices",
        "ai_engine": "Multi-agent orchestration with LangChain + AutoGen",
        "vector_storage": "Pinecone/Weaviate integration",
        "communication": "gRPC + REST",
        "services": list(SERVICES.keys()),
        "features": [
            "Dynamic AI Kernel",
            "Conversational AI Copilot",
            "Enhanced CV Engine",
            "Analytics Brain",
            "Automation Intelligence",
            "Computer Vision AI",
            "Plugin Framework"
        ]
    }

@app.get("/api/v2/services")
async def list_services():
    """List all available microservices"""
    return {
        "services": {
            name: {
                "description": get_service_description(name),
                "grpc_port": config["grpc_port"],
                "http_port": config["http_port"],
                "status": "configured"
            }
            for name, config in SERVICES.items()
        }
    }

def get_service_description(service_name: str) -> str:
    """Get description for a service"""
    descriptions = {
        "ai-kernel": "Central AI orchestration and reasoning engine",
        "identity": "Authentication and user management",
        "cv-engine": "Extended CV and portfolio generation",
        "conversational": "AI Copilot and chat functionality",
        "analytics": "Data processing and insights",
        "automation": "Workflow and network automation",
        "vision": "Computer vision and media processing",
        "plugin": "Plugin management and extensibility"
    }
    return descriptions.get(service_name, "AI Fusion Core microservice")

# Exception handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTP exceptions"""
    logger.warning(
        "HTTP exception",
        status_code=exc.status_code,
        detail=exc.detail,
        path=request.url.path,
        method=request.method
    )
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail}
    )

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler"""
    logger.error(
        "Unhandled exception",
        exc_info=exc,
        path=request.url.path,
        method=request.method
    )
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"}
    )

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
        log_level=settings.LOG_LEVEL.lower(),
        access_log=True
    )