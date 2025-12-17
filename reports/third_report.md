📈 Confronto tra training breve (3 epoche) e training esteso con early stopping (15 epoche)
🔹 1. Andamento della loss (training log)
Epoca	Loss (3 epoche)	Loss (early stopping)
1	1.4828	1.5526
2	0.7841	0.7610
3	0.6486	0.6137
4–10	—	↓ fino a 0.4145
11–15	—	plateau / stop

Il secondo training ha mostrato:

Trend di convergenza più esteso, fino a una loss finale di 0.4145

Stabilità senza oscillazioni significative

Early stopping efficace dopo 15 epoche, con salvataggio del modello al punto di minima loss (epoca 10)

🔹 2. Risultati di retrieval cross-modale (Text→Pose)
Metriche	3 epoche	Early stopping
Recall@1	0.009 (0.9%)	0.027 (2.7%)
Recall@5	0.053 (5.3%)	0.138 (13.8%)
Mean Average Prec.	0.047	0.115

📌 Osservazioni:

Il nuovo modello recupera il match corretto nel top 5 in oltre 13% dei casi → 2.6× meglio rispetto al primo esperimento.

mAP più che raddoppiata, indice di miglior generalizzazione nell’intero ranking.

Il Recall@1 rimane basso, ma il trend è in crescita.

🔹 3. Visualizzazione t-SNE (embedding pose e testo)
T-SNE	3 Epoche	15 Epoche
Cluster visibili?	Parziali	✅ Maggior definizione
Mixing pose-testo	Scarso	✅ Più mescolanza nei cluster
Outlier testuali	Presenti	↘ Ridotti
Varianza spiegata	40%	Da ricalcolare

La visualizzazione aggiornata mostra cluster più compatti, overlap maggiore tra le due modalità (pose ↔ testo) e minore dispersione. Questo conferma che il training esteso ha migliorato la coerenza semantica dello spazio latente.

✅ Conclusioni

L’utilizzo di early stopping e un training più prolungato ha permesso di:

Ottenere embedding più allineate e strutturate

Migliorare notevolmente le metriche di retrieval

Ridurre la loss contrastiva in modo stabile

Questo esperimento dimostra che il modello dual encoder è promettente, ma la qualità dello spazio latente è ancora influenzata da:

Il numero di epoche

La selezione dei negativi (random per ora)

La ricchezza semantica delle frasi testuali


Prossimi step:
🔜 Prossimi step

Aggiungere validation set separato con evaluation periodica

Provare una loss contrastiva con hard negative mining

Aumentare la batch size e/o testare temperature diverse

Valutare data augmentation lato testo e pose