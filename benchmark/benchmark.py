from __future__ import annotations

import json
import math
import re
import statistics
import time
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

DATASET = Path(__file__).with_name("dataset.json")
RESULTS = Path(__file__).with_name("results")
RESULTS.mkdir(exist_ok=True)
STOP = set("a an and are as at be been but by can could did do does for from how i if in into is it its me of on or our please should that the their them there these this to was were what when where which who why with would you your".split())
TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")

def tokens(text: str) -> list[str]: return [x.lower() for x in TOKEN_RE.findall(text or "")]
def content(text: str) -> list[str]: return [x for x in tokens(text) if x not in STOP and len(x) > 1]
def norm(values):
    m=max(values) if values else 0.0
    return [x/m if m else 0.0 for x in values]

def bm25(query, docs):
    q=content(query); fs=[]; dfs={}; lengths=[]
    for d in docs:
        f={}
        for t in content(d): f[t]=f.get(t,0)+1
        fs.append(f); lengths.append(sum(f.values()))
        for t in f: dfs[t]=dfs.get(t,0)+1
    avg=sum(lengths)/max(1,len(lengths)); out=[]
    for f,dl in zip(fs,lengths):
        s=0.0
        for t in q:
            if t not in f: continue
            idf=math.log(1+(len(docs)-dfs[t]+.5)/(dfs[t]+.5))
            s+=idf*(f[t]*2.5)/(f[t]+1.5*(.25+.75*dl/max(avg,1)))
        out.append(s)
    return out

def rank_hybrid(query, docs):
    b=norm(bm25(query,docs)); v=TfidfVectorizer(stop_words="english",ngram_range=(1,2),min_df=1); m=v.fit_transform(docs); t=norm(cosine_similarity(v.transform([query]),m)[0].tolist())
    return sorted(range(len(docs)),key=lambda i:(-(.55*b[i]+.45*t[i]),i))

def rerank(query, docs, base_rank):
    q=set(content(query)); qi=query.lower(); out=[]
    for rank,i in enumerate(base_rank):
        text=docs[i]; low=text.lower(); terms=set(content(text)); overlap=len(q&terms)/max(1,len(q)); phrase=sum(1 for x in q if x in low)/max(1,len(q)); score=.45*(1/(rank+1))+.35*overlap+.20*phrase
        if re.search(r"\b(what is|what are|define|defined)\b",qi) and re.search(r"\b(is|are|defined as|refers to|denotes)\b",low): score+=.18
        if re.search(r"\b(how|calculate|formula)\b",qi) and re.search(r"\b(calculate|computed|formula|equation|using)\b",low): score+=.12
        out.append((score,i))
    return [i for _,i in sorted(out,key=lambda x:(-x[0],x[1]))]

def hit(ret,rel,k): return float(bool(set(ret[:k])&rel))
def recall(ret,rel,k): return len(set(ret[:k])&rel)/max(1,len(rel))
def mrr(ret,rel,k):
    for r,i in enumerate(ret[:k],1):
        if i in rel:return 1/r
    return 0.0

def ndcg(ret,rel,k):
    dcg=sum(1/math.log2(r+1) for r,i in enumerate(ret[:k],1) if i in rel); ideal=sum(1/math.log2(r+1) for r in range(1,min(k,len(rel))+1)); return dcg/ideal if ideal else 0.0

def evaluate(name, ranker, docs, queries, k_values=(1,3,5)):
    series={f"Hit@{k}":[] for k in k_values}; series.update({f"Recall@{k}":[] for k in k_values}); series.update({f"NDCG@{k}":[] for k in k_values}); series["MRR"]=[]; series["latency_ms"]=[]
    for item in queries:
        rel={next(i for i,d in enumerate(docs) if d["id"]==rid) for rid in item["relevant"]}; t=time.perf_counter(); ret=ranker(item["query"]); series["latency_ms"].append((time.perf_counter()-t)*1000); series["MRR"].append(mrr(ret,rel,max(k_values)))
        for k in k_values: series[f"Hit@{k}"].append(hit(ret,rel,k)); series[f"Recall@{k}"].append(recall(ret,rel,k)); series[f"NDCG@{k}"].append(ndcg(ret,rel,k))
    result={"retriever":name,"queries":len(queries)}
    for key,vals in series.items(): result[key]=round(statistics.mean(vals),4)
    result["latency_p95_ms"]=round(sorted(series["latency_ms"])[max(0,math.ceil(len(series["latency_ms"])*.95)-1)],4)
    return result

def main():
    data=json.loads(DATASET.read_text(encoding="utf-8")); docs=data["documents"]; queries=data["queries"]; texts=[d["text"] for d in docs]
    hybrid=lambda q:rank_hybrid(q,texts); reranked=lambda q:rerank(q,texts,rank_hybrid(q,texts))
    results=[evaluate("Hybrid",hybrid,docs,queries),evaluate("Hybrid+Rerank",reranked,docs,queries)]
    stage=[]
    for item in queries:
        retrieval=[]; reranking=[]
        for _ in range(3):
            t=time.perf_counter(); base=rank_hybrid(item["query"],texts); retrieval.append((time.perf_counter()-t)*1000)
            t=time.perf_counter(); rerank(item["query"],texts,base); reranking.append((time.perf_counter()-t)*1000)
        stage.append({"retrieval_ms":statistics.mean(retrieval),"rerank_ms":statistics.mean(reranking)})
    latency={"retrieval_mean_ms":round(statistics.mean(x["retrieval_ms"] for x in stage),4),"retrieval_p95_ms":round(sorted(x["retrieval_ms"] for x in stage)[max(0,math.ceil(len(stage)*.95)-1)],4),"rerank_mean_ms":round(statistics.mean(x["rerank_ms"] for x in stage),4),"rerank_p95_ms":round(sorted(x["rerank_ms"] for x in stage)[max(0,math.ceil(len(stage)*.95)-1)],4)}
    output={"dataset":str(DATASET),"results":results,"pipeline_latency":latency,"note":"Generated measurements; rerun after retrieval/chunking changes. This benchmark uses the committed retrieval/reranking implementation, not hardcoded claims."}
    (RESULTS/"benchmark_results.json").write_text(json.dumps(output,indent=2),encoding="utf-8")
    lines=["# Retrieval Benchmark Results","","Generated by `python benchmark/benchmark.py`.","","| Retriever | Hit@1 | Hit@3 | Recall@5 | MRR | NDCG@5 | Mean latency (ms) | p95 latency (ms) |","|---|---:|---:|---:|---:|---:|---:|---:|"]
    for r in results: lines.append(f"| {r['retriever']} | {r['Hit@1']:.3f} | {r['Hit@3']:.3f} | {r['Recall@5']:.3f} | {r['MRR']:.3f} | {r['NDCG@5']:.3f} | {r['latency_ms']:.3f} | {r['latency_p95_ms']:.3f} |")
    lines += ["","## Pipeline latency","",f"- Retrieval mean: {latency['retrieval_mean_ms']:.3f} ms",f"- Retrieval p95: {latency['retrieval_p95_ms']:.3f} ms",f"- Rerank mean: {latency['rerank_mean_ms']:.3f} ms",f"- Rerank p95: {latency['rerank_p95_ms']:.3f} ms"]
    (RESULTS/"benchmark_report.md").write_text("\n".join(lines)+"\n",encoding="utf-8"); print("\n".join(lines))

if __name__=="__main__": main()
