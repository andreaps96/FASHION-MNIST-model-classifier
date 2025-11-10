
# Classificazione di Capi di Abbigliamento (Fashion-MNIST) con CNN in PyTorch

## Panoramica

Questo progetto implementa una **Convolutional Neural Network (CNN)** utilizzando PyTorch per classificare immagini di capi di abbigliamento provenienti dal dataset **Fashion-MNIST**, un benchmark largamente utilizzato come alternativa più complessa al classico MNIST.

Il notebook segue un approccio completo e sistematico, affrontando tutte le fasi essenziali di sviluppo di un modello di deep learning: dalla preparazione dei dati e la loro augmentazione, alla definizione di un’architettura CNN robusta e regolarizzata, fino all’adozione di strategie di training avanzate come il **Learning Rate Finder**, lo scheduler **OneCycleLR** e l’**Early Stopping** per ottenere un bilanciamento ottimale tra accuratezza e generalizzazione.

---

## Funzionalità Principali

-   **Preparazione dei Dati**: Caricamento del dataset Fashion-MNIST e suddivisione in set di training (80%) e validazione (20%).
-   **Data Augmentation**: Applicazione di trasformazioni per migliorare la capacità del modello di generalizzare su nuovi campioni, tra cui:
    -   Rotazioni casuali (`RandomRotation`)
    -   Traslazioni e variazioni di scala (`RandomAffine`)
    -   Distorsioni prospettiche (`RandomPerspective`)
-   **Architettura CNN Personalizzata**:
    -   Tre blocchi convoluzionali basati su `Conv2d`, `BatchNorm2d`, `ReLU` e `Dropout`.
    -   Calcolo dinamico della dimensione dell’input per il classificatore, garantendo flessibilità all’architettura.
    -   Inizializzazione dei pesi con schemi **Xavier** e **Kaiming** per una migliore stabilità e convergenza durante l’addestramento.
-   **Ricerca del Learning Rate Ottimale**: Utilizzo della libreria `torch-lr-finder` per determinare il valore di learning rate più efficace prima dell’avvio del training.
-   **Strategia di Training Avanzata**:
    -   Ottimizzatore **AdamW** con weight decay disaccoppiato per una migliore regolarizzazione.
    -   Scheduler **OneCycleLR** per la gestione dinamica di learning rate e momentum, accelerando la convergenza.
    -   **Early Stopping** per interrompere automaticamente il training quando le metriche di validazione non migliorano, salvando il modello con le migliori performance.
-   **Valutazione del Modello**: Analisi delle performance finali tramite:
    -   **Classification Report** con metriche dettagliate di precision, recall e F1-score per ogni classe.
    -   **Matrice di Confusione** per analizzare gli errori di classificazione tra i diversi tipi di capi di abbigliamento.

---

## Architettura del Modello

Il modello si compone di due moduli principali: un estrattore di feature convoluzionale e un classificatore fully connected.

1.  **Blocchi Convoluzionali**:
    -   **Conv Block 1**: `Conv2d(1, 32)` → `BatchNorm2d` → `ReLU` → `Dropout(0.2)`
    -   **Conv Block 2**: `Conv2d(32, 64)` → `BatchNorm2d` → `ReLU` → `Dropout(0.4)`
    -   **Conv Block 3**: `Conv2d(64, 128)` → `BatchNorm2d` → `ReLU` → `Dropout(0.4)`

2.  **Classificatore**:
    -   `Flatten`
    -   `Linear(input_dinamico, 128)` → `ReLU` → `Dropout(0.4)`
    -   `Linear(128, 10)` (10 classi, una per ogni categoria del dataset Fashion-MNIST)

---

## Risultati

L’esecuzione del notebook produce i seguenti output:

-   Visualizzazione del **Learning Rate Finder** per l’identificazione del learning rate ottimale.
-   Log dettagliati di training e validazione per ogni epoca, con l’andamento di loss e accuracy.
-   Grafici delle curve di apprendimento (loss e accuracy) per training e validation.
-   Un **report di classificazione** con le metriche per ciascuna delle 10 categorie (T-shirt/top, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, Ankle Boot).
-   Una **matrice di confusione** che evidenzia le principali aree di confusione, come ad esempio tra “Shirt” e “T-shirt/top”.