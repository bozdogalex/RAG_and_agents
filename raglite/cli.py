# raglite/cli.py
import json
from pathlib import Path
import typer
from . import pipeline
from .config import settings

app = typer.Typer(add_completion=False)

@app.command()
def build(pdf: Path):
    chunks = pipeline.build_corpus(str(pdf))
    vecs = pipeline.embed_corpus(chunks, settings.embed_model)
    out = {"chunks": chunks, "vectors": vecs}
    Path("artifacts").mkdir(exist_ok=True)
    Path("artifacts/corpus.json").write_text(json.dumps(out))
    typer.echo("Saved artifacts/corpus.json")

@app.command()
def ask(query: str, k: int = settings.top_k):
    data = json.loads(Path("artifacts/corpus.json").read_text())
    chunks, vectors = data["chunks"], data["vectors"]
    ctxs = pipeline.retrieve(query, chunks, vectors, k)
    ans = pipeline.answer(query, ctxs, settings.chat_model)
    typer.echo(ans)

@app.command()
def eval(valfile: Path = Path("data/val.json")):
    data = json.loads(valfile.read_text())
    query = data[0]["question"]
    true_ans = data[0]["ideal_answer"]
    corpus = json.loads(Path("artifacts/corpus.json").read_text())
    ctxs = pipeline.retrieve(query, corpus["chunks"], corpus["vectors"], k=2)
    ans = pipeline.answer(query, ctxs, settings.chat_model)
    from .evaluate import score
    s = score(query, ans, true_ans, settings.chat_model)
    typer.echo(s)

if __name__ == "__main__":
    app()
