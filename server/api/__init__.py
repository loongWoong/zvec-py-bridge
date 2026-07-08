"""API routers for the zvec REST bridge."""
from api.admin import router as admin_router  # noqa: F401
from api.collection import router as collection_router  # noqa: F401
from api.document import router as document_router  # noqa: F401
from api.embedding import router as embedding_router  # noqa: F401
from api.index import router as index_router  # noqa: F401
from api.query import router as query_router  # noqa: F401
