"""FastAPI application."""

from hcmaic.api.app import create_app
from hcmaic.api.dual_app import create_dual_app
from hcmaic.api.groundtruth_review import create_groundtruth_review_app

__all__ = ["create_app", "create_dual_app", "create_groundtruth_review_app"]
