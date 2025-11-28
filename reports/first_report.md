Primo report di avanzamento

In questa prima fase mi sono concentrato su un dataset relativamente compatto ma molto specifico per il problema della tesi: EC3D (Exercise Correction 3D). L’obiettivo non era ancora costruire il modello finale, ma impostare una pipeline pulita e riproducibile per la classificazione degli errori a partire da pose 3D, e fare un primo esperimento preliminare di allineamento pose↔testo in stile CLIP. Questo mi permette di avere sia risultati quantitativi concreti, sia una base “solida ma leggera” su cui motivare il passaggio a dataset più ricchi e modelli multimodali più espressivi.

EC3D: ricostruzione delle sequenze e delle etichette

EC3D contiene tre esercizi fondamentali – squat, affondi (lunges) e plank – eseguiti da quattro soggetti, sia in forma corretta sia con errori tipici. Gli autori rilasciano un file data_3D.pickle con tutte le pose 3D e un array di etichette testuali “compattate”. A partire dal codice della loro repository ho ricostruito il significato di queste etichette: per ogni frame sono indicati l’esercizio (SQUAT, Lunges, Plank), il soggetto (Hugues, Sena, Isinsu, Vidit), un instruction id intero che codifica il tipo di esecuzione (corretta o con errore) e gli indici di sequenza/frame.

Utilizzando la mappa di label proposta dagli autori ho collegato ogni instruction id a un nome semantico (“Correct”, “Feets too wide”, “Knees inward”, “Not low enough”, “Banana back”, “Rolled back”, ecc.) e ho verificato che la distribuzione delle combinazioni esercizio/soggetto/errore ottenuta dal mio preprocessing coincidesse con quella riportata nel paper. Da qui ho ricostruito le sequenze temporali vere e proprie: raggruppando i frame con stesso esercizio, soggetto, instruction id e id di prova, ordinati per tempo, ottengo per ogni ripetizione una sequenza di forma (T, 3, 25) (T frame, 3 coordinate, 25 joint).

Per ogni sequenza ho definito anche un’etichetta globale intera da 0 a 11 che rappresenta la coppia esercizio–errore (es. “SQUAT – Correct”, “Lunges – Knees pass toes”, “Plank – Banana back”). In totale ho ottenuto 371 sequenze con metadati strutturati (soggetto, esercizio, tipo di errore, numero di frame, ecc.), salvate in ec3d_sequences.pkl, e uno split cross–subject (split_cross_subject.json) in cui tre soggetti (Hugues, Sena, Isinsu) sono usati per il training e il quarto (Vidit) per il test. Questo garantisce che le valutazioni misurino davvero la capacità di generalizzare a persone nuove.

Baseline di classificazione degli errori

Come primo passo ho costruito una baseline “classica” su feature statiche estratte dalle sequenze di pose. Dopo aver centrato lo scheletro sul bacino per rendere le pose confrontabili tra soggetti, ho riorganizzato ogni sequenza in (T, 25, 3) e calcolato per ogni giunto e coordinata tre statistiche lungo il tempo: media, deviazione standard e ampiezza (max–min). Concatenando tutte queste statistiche ottengo un vettore di 225 feature per sequenza, che cattura in modo grossolano quanto si muove ogni articolazione, quanto è stabile e su che range di movimento si colloca.

Su queste feature ho addestrato un classificatore Random Forest per predire la classe esercizio–errore (11 classi, escluso il caso “Unknown” poco rappresentato). Sul soggetto di test (Vidit), la Baseline 1 raggiunge un’accuracy di circa 0,74 e una macro–F1 di circa 0,68. Dalla matrice di confusione emerge che il modello riconosce bene gli errori più marcati (“Feets too wide”, “Knees pass toes”, “Banana back”, “Rolled back”), mentre fa più fatica sui casi borderline, cioè nel separare esecuzioni corrette da errori sottili (schiena leggermente flessa, profondità appena insufficiente, ecc.). Questo è coerente con il tipo di feature usate: sono globali e non descrivono esplicitamente la dinamica fine del gesto.

Per avvicinarmi alla logica dei sistemi tipo AI-Fit ho implementato una seconda baseline basata sul confronto con una “firma corretta” per ogni esercizio. Per ciascun esercizio ho calcolato una firma s_ref come media delle feature delle sole sequenze corrette nel train; per ogni ripetizione ho poi costruito un vettore di feature come differenza assoluta rispetto alla s_ref del relativo esercizio. Questo nuovo spazio descrive esplicitamente “quanto questa ripetizione si discosta dal modo in cui il coach esegue il movimento”. Anche qui ho addestrato una Random Forest (Baseline 2). Le prestazioni sul test restano buone ma leggermente inferiori (accuracy ≈ 0,69, macro–F1 ≈ 0,67): l’idea del confronto con la firma è concettualmente interessante, ma con feature molto semplici tende ad appiattire alcune differenze tra errori diversi.

Primo prototipo contrastivo pose↔testo

Per collegare le pose a un possibile feedback testuale ho implementato un primo prototipo di modello contrastivo pose↔testo ispirato a CLIP, consapevolmente minimale. A partire dalle 11 classi esercizio–errore ho definito per ciascuna una breve descrizione in inglese che simula il commento di un coach (ad es. “Your knees are caving inward. Gently push them outward to keep them aligned with your feet.” per “Knees inward”).

Ho costruito:
	•	un piccolo encoder per le pose (MLP che mappa le 225 feature in un embedding a bassa dimensione),
	•	un encoder testuale minimale (un embedding vettoriale per classe, quindi senza sfruttare ancora un vero modello linguistico).

Il modello è addestrato con una loss contrastiva simmetrica: all’interno di ogni batch gli embedding pose e testo delle coppie corrette devono risultare più vicini rispetto alle coppie sbagliate. In 20 epoche di training la loss scende in modo regolare, ma la retrieval accuracy top–1 sul soggetto di test resta bassa (circa 5–9%, contro ~1% random). Il segnale non è nullo, ma l’allineamento pose↔testo ottenuto con così pochi dati, feature molto povere e un encoder testuale quasi banale non è ancora sufficiente per un sistema di feedback affidabile. Questo risultato, però, è utile proprio perché mette in evidenza quali ingredienti mancano: più dati, testi più ricchi e un encoder linguistico serio.

Lezioni imparate e passi successivi

Questa prima fase su EC3D mi ha dato tre cose importanti:
	1.	Una pipeline dati robusta, con sequenze 3D pulite, metadati chiari e split cross–subject riutilizzabile per tutti gli esperimenti successivi.
	2.	Baseline interpretabili per la classificazione degli errori, che mostrano come già semplici statistiche globali permettano di riconoscere molti errori macroscopici, ma evidenziano anche i limiti sui casi sottili.
	3.	Un primo banco di prova multimodale: il mini–CLIP dimostra che l’idea di allineare pose e testo è realizzabile tecnicamente, ma richiede dataset e modelli più ricchi per diventare davvero efficace.

Dato che il dataset FLEX, pensato esattamente per il feedback correttivo pose↔testo, al momento non è ancora accessibile, il passo successivo naturale è sfruttare meglio i dataset che sono già ottenibili:
	•	FLAG3D, che offre molti esercizi con istruzioni testuali dettagliate sulla tecnica corretta;
	•	Fit3D, che fornisce video multi-view e pose 3D ad alta qualità su 37 esercizi, ideale per pre-addestrare un encoder di pose e sperimentare la logica di “signature comparison” tipo AI-Fit;
	•	Fitness-AQA, che, pur non avendo pose 3D di ground truth, contiene video “in the wild” annotati con errori multipli e punteggi di qualità, utili per valutare la robustezza dei modelli in scenari realistici.

L’idea per la fase successiva della tesi è quindi di usare EC3D (ed eventualmente Waseda) come banco di prova finale sugli errori, ma di appoggiarsi a questi dataset più ampi per addestrare un encoder pose↔testo più espressivo, e poi verificare se il pretraining multimodale aiuta davvero a migliorare la classificazione degli errori e la qualità del feedback rispetto alle baseline costruite in questa prima parte del lavoro.
