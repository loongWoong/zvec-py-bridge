"""Service layer: orchestrates manager + mapper for each domain."""
from service.admin_service import AdminService  # noqa: F401
from service.collection_service import CollectionService  # noqa: F401
from service.document_service import DocumentService  # noqa: F401
from service.embedding_service import EmbeddingService  # noqa: F401
from service.index_service import IndexService  # noqa: F401
from service.query_service import QueryService  # noqa: F401

# Process-wide singletons wired to the shared manager.
collection_service = CollectionService()
document_service = DocumentService()
embedding_service = EmbeddingService()
query_service = QueryService()
index_service = IndexService()
admin_service = AdminService()
