🔍 Valutazione del modello dopo GradNorm
📈 Metriche quantitative (retrieval)
Metrica	Text → Pose	Pose → Text	Media (%)
R@1	2.4%	2.7%	2.5% 🔽
R@5	12.1%	11.7%	11.9% 🔽
R@10	24.2%	23.5%	23.8% 🔽
mAP	10.0%	10.1%	10.1% ↔️
MRR	10.0%	10.1%	10.1% ↔️
Mean Rank	~39	~40	~39.6 🔽

🔎 Analisi:

I valori sono coerenti con il comportamento in loss: migliori delle versioni con forte overfitting, ma ancora bassi in assoluto.

La rank@1 è molto bassa. Questo può significare:

Scarsa discriminabilità tra le azioni simili

Latenti distribuiti ma non separati semanticamente

Margine nella loss troppo stretto

🎯 Visualizzazione t-SNE

L'immagine è molto utile per confermare il comportamento della loss:

Si vedono cluster separati, ma anche molta sovrapposizione.

Alcune azioni formano gruppi molto densi e puri (ottimo segnale).

Altre sono miste o spezzate, indicando che:

o il testo non è discriminativo abbastanza,

o l’allineamento testo-posa è ancora approssimativo,

o le pose stesse hanno ambiguità strutturali.

🧠 Cosa ha funzionato (efficace)

✅ GradNorm ha stabilizzato il training
✅ Ha portato a un learning più bilanciato tra i due encoder
✅ Ha rallentato l’overfitting che affliggeva i run precedenti
✅ Embedding visivamente più organizzati nel t-SNE

⚠️ Cosa ancora non funziona (limiti)

🔸 Le metriche di retrieval sono ancora basse
🔸 Alcune azioni sono sovrapposte visivamente
🔸 Il modello non è ancora fortemente contrastivo
🔸 Potremmo essere limitati dalla capacità del text encoder (DistilBERT è leggero)

✅ Prossimi step consigliati

Aggiungere un margine alla NT-Xent loss
→ attualmente usi similarity / temperature, ma senza un margin, rischi che i negativi troppo vicini non vengano penalizzati abbastanza.

Testare cross-modal mining dei negativi
→ includere solo i negativi più vicini nel batch per aumentare la spinta del gradiente.

Warmup learning rate + scheduler a step
→ la curva piatta della train loss suggerisce che potrebbe servire un picco iniziale più graduale.

Prossimo test: sostituire DistilBERT con MiniLM o MPNet (più informativi)
→ per capire se il collo di bottiglia è nel testo.

Aggiungere un classificatore ausiliario sull’embedding pose/text (multi-task)
→ come proxy task di classificazione su 7–10 gruppi di azioni. Aiuta a strutturare la latente.