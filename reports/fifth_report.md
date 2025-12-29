| Setup                        | Recall@1 | mAP       | Note                       |
| ---------------------------- | -------- | --------- | -------------------------- |
| Fine-tuning (I001–I006)      | **0.94** | **0.956** | Upper bound                |
| Zero-shot (template base)    | 0.41     | 0.558     | Peggior caso               |
| Zero-shot (template verbose) | 0.54     | 0.680     | Più naturale → meglio      |
| One-shot (pose ↔ pose)       | 0.66     | 0.765     | Nessun testo, ma 1 esempio |


📌 Cosa funziona

Il modello apprende bene da EC3D se fine-tuned.

Le frasi verbose migliorano lo zero-shot.

Il retrieval one-shot è promettente: l’architettura si adatta bene anche a una configurazione minima.