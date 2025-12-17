### Analisi qualitativa e quantitativa dello spazio latente condiviso

#### 1. Metriche di retrieval cross-modale

Dopo l'addestramento del modello dual encoder su 3 epoche, è stata effettuata una valutazione quantitativa sulla qualità dello spazio latente condiviso tra rappresentazioni testuali e rappresentazioni posturali. L'obiettivo era misurare la capacità del sistema di associare correttamente coppie testo-posa all'interno dello spazio latente, utilizzando metodi di retrieval cross-modale.

I risultati ottenuti sul validation set (2160 coppie) sono stati i seguenti:

* **Recall@1**: 0.009 (0.9%)
* **Recall@5**: 0.053 (5.3%)
* **Mean Average Precision (mAP)**: 0.047 (4.7%)

Tali risultati indicano che, allo stato attuale, il modello riesce raramente a collocare il match corretto tra i primi risultati restituiti nella fase di retrieval. Sebbene il Recall@5 mostri un leggero miglioramento rispetto al Recall@1, la performance globale suggerisce che lo spazio latente condiviso appreso non è ancora sufficientemente discriminativo o allineato. Questo può dipendere da diversi fattori:

* Complessità semantica delle descrizioni testuali rispetto alla rappresentazione pose-based
* Limitato numero di epoche (solo 3) nel training iniziale
* Squilibrio nei dati o alta similarità semantica tra azioni diverse (confondibilità)

#### 2. Analisi visiva con t-SNE

A complemento delle metriche numeriche, è stata effettuata un'analisi qualitativa tramite visualizzazione 2D con **t-SNE**, una tecnica non lineare di riduzione dimensionale che preserva la struttura locale dei dati. Le embedding testuali e posturali sono state proiettate con t-SNE e colorate in base all'action class.

![](t-SNE%20first%20try.png)

Dalla distribuzione emerge quanto segue:

* **Presenza di cluster ben definiti** per alcune classi di azioni (es. Azione 0, Azione 3, Azione 7), suggerendo che il modello ha imparato a separare alcune categorie in modo coerente.
* **Forte sovrapposizione tra modalità**: in molti cluster i punti relativi a "pose" e "testo" risultano ancora distanti o poco mischiati. Questo indica che l'allineamento multimodale non è ancora efficace: lo spazio latente contiene ancora regioni separate per le due modalità.
* **Numerosi outlier**, soprattutto per le embedding testuali (triangoli), che si distribuiscono sparse e distanti dai cluster principali. Ciò potrebbe indicare instabilità nella rappresentazione testuale o presenza di testi semanticamente ambigui o generici.

#### 3. Varianza spiegata con PCA

L'analisi tramite PCA ha mostrato che i primi due componenti principali spiegano circa il 40% della varianza totale (30.2% + 9.7%). Questo conferma che lo spazio latente ha una struttura dispersa, e che la maggior parte dell'informazione è distribuita su più dimensioni, rendendo difficile la separazione lineare tra classi.

#### 4. Conclusioni preliminari

Questo primo esperimento conferma la validità della pipeline tecnica, mostrando che il training dual encoder produce uno spazio latente strutturato. Tuttavia, i risultati quantitativi e qualitativi evidenziano margini di miglioramento significativi, soprattutto nell'allineamento tra le due modalità. Sarà pertanto necessario:

* Aumentare il numero di epoche di addestramento
* Bilanciare meglio i dati o introdurre strategie di hard negative mining
* Migliorare l'adattamento dei tokenizer/testo per enfatizzare concetti chiave rilevanti per il matching
* Valutare tecniche alternative di loss contrastiva o introduzione di supervisioni più forti

Questi interventi saranno oggetto degli esperimenti successivi.
