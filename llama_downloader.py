from llama_index import download_loader

ChromaReader = download_loader("ChromaReader", refresh_cache=True)
MarkdownReader = download_loader("MarkdownReader", refresh_cache=True)
Github = download_loader("GithubRepositoryReader", refresh_cache=True)
Unstructured = download_loader("UnstructuredReader", refresh_cache=True)