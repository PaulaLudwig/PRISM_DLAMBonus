# PRISM_DLAMBonus — Übergabe-Checkliste

**Deadline: heute, 04.09.2026, 23:59 CEST**

Stand: `main`-Branch enthält die finale PRISM-Abgabe (`submission_template/`), bereit zum Packen und Hochladen. Der Branch `submission-with-lstm` enthält eine funktionierende LSTM-Alternative als Backup (nicht die finale Wahl, siehe Hinweis unten).

---

## 1. Submission Template testen

Bevor irgendetwas hochgeladen wird: einmal komplett lokal durchtesten, dass `predict.py` mit dem privaten Single-File-Contract funktioniert (also genau so, wie es später bei der Bewertung laufen wird).

Alle Befehle im Projekt-Root ausführen (`cd` in `PRISM_DLAMBonus`), auf Branch `main`.

**1.1 Test-Fixture erzeugen** (maskiert die letzten 336 Stunden von `train.csv`, simuliert also den privaten Test-Input):

```bash
mkdir -p data/_predict_test
python3 - << 'EOF'
import pandas as pd
import numpy as np

HORIZON = 336
df = pd.read_csv("data/train.csv", parse_dates=["timestamp"])
df = df.sort_values(["series_id", "timestamp"]).reset_index(drop=True)

test_rows = []
index_rows = []
for sid, g in df.groupby("series_id"):
    g = g.sort_values("timestamp").reset_index(drop=True).copy()
    mask_start = len(g) - HORIZON
    masked_timestamps = g.loc[mask_start:, "timestamp"]
    g.loc[mask_start:, "target"] = np.nan
    test_rows.append(g)
    index_rows.append(pd.DataFrame({"series_id": sid, "timestamp": masked_timestamps}))

test_input = pd.concat(test_rows, ignore_index=True)
forecast_index = pd.concat(index_rows, ignore_index=True)

test_input.to_csv("data/_predict_test/test_input.csv", index=False)
forecast_index.to_csv("data/_predict_test/forecast_index_test.csv", index=False)
print("test_input rows:", len(test_input))
print("forecast_index rows:", len(forecast_index))
EOF
```

**1.2 `predict.py` laufen lassen** (genau der Befehl, den die Bewertung später verwendet, nur mit lokalem Pfad statt `/data/input` etc.):

```bash
cd submission_template
python3 predict.py --input_dir ../data/_predict_test --output_file /tmp/out_prism.csv --checkpoint checkpoint.pt
cd ..
```

Erwartete Ausgabe: `wrote 32256 rows to /tmp/out_prism.csv`. Kommt ein Fehler (z. B. `ModuleNotFoundError`, `FileNotFoundError`), stimmt etwas an den Pfaden/Dateien im `submission_template/`-Ordner nicht — vor dem Weitermachen fixen.

**1.3 WAPE gegen die echten (maskierten) Werte prüfen** (Erwartung: ca. 0.2076, wie im Report berichtet):

```bash
python3 - << 'EOF'
import pandas as pd

pred = pd.read_csv("/tmp/out_prism.csv", parse_dates=["timestamp"])
truth = pd.read_csv("data/train.csv", parse_dates=["timestamp"])

merged = pred.merge(truth[["series_id","timestamp","target"]], on=["series_id","timestamp"], how="left")
missing = merged["target"].isna().sum()
print("matched rows:", len(merged) - missing, "of", len(merged))

wape = (merged["target"] - merged["prediction"]).abs().sum() / merged["target"].abs().sum()
print("WAPE:", round(wape, 4))
EOF
```

Wenn `matched rows` nicht `32256 of 32256` ist oder die WAPE stark abweicht: nicht weitermachen, sondern erst klären woran es liegt.

**1.4 Aufräumen** (Test-Fixture wird nicht mit abgegeben):

```bash
rm -rf data/_predict_test
```

---

## 2. Restliche Schritte bis zur Abgabe

1. **`final_submission.zip` packen** (aus `submission_template/` heraus, nachdem Schritt 1 erfolgreich war):
   ```bash
   cd submission_template
   zip -r final_submission.zip predict.py requirements.txt checkpoint.pt scaler.npz src
   cd ..
   ```
   Wichtig: `scaler.npz` muss mit rein (enthält die Normalisierungsstatistiken, ohne die `predict.py` fehlschlägt) — steht auch so in der README.

2. **Zip-Inhalt kurz gegenchecken**, bevor hochgeladen wird:
   ```bash
   unzip -l submission_template/final_submission.zip
   ```
   Sollte enthalten: `predict.py`, `requirements.txt`, `checkpoint.pt`, `scaler.npz`, `src/data.py`, `src/prism.py`.

3. **Auf die Hugging-Face-Leaderboard-Space hochladen**:
   https://huggingface.co/spaces/AIML-TUDA/dlam-ts-project-leaderboard-2026
   Mit HF-Account einloggen, `final_submission.zip` als finales Model-Archiv hochladen. Das ist die eigentliche, bewertungsrelevante Abgabe.

4. **Report-PDF bei Moodle hochladen** — aktuelle Version enthält bereits:
   - Method-Abschnitt zum Exposé-Vergleich
   - Figur 1 (Architektur-Diagramm)
   - vollständige Conclusion
   - Contributions für alle 4 Mitglieder
   - Hinweis, dass das LSTM-Ergebnis (0.1357) nicht reproduzierbar war (echter Retrain ergab ~0.30)

5. **Code/Repository bei Moodle einreichen** — laut Aufgabenstellung entweder als Zip-Datei oder als Link zum GitHub-Repo (`main`-Branch), beides ist ausdrücklich erlaubt ("Submit your code as a well-structured repository (e.g., a zip file or a link to a Git repository)"). Falls ihr euch für die Zip-Variante entscheidet: aus dem Projekt-Root heraus packen, aber `data/`, `__pycache__/`, `.git/` und sonstige große/generierte Ordner ausschließen, z. B.:
   ```bash
   zip -r PRISM_DLAMBonus_code.zip . -x "data/*" "*__pycache__*" ".git/*" "*.DS_Store"
   ```

---

## Hinweis zur Modellwahl

`main` enthält **PRISM** (patch-preserving bridge, WAPE 0.2076) als finale Abgabe. Das LSTM-Baseline-Ergebnis aus dem Report (WAPE 0.1357) war bei einem echten Retrain **nicht reproduzierbar** (ergab WAPE ~0.30, schlechter als PRISM) — deshalb wurde PRISM als sicherere, tatsächlich verifizierte Wahl für die Leaderboard-Abgabe genommen. Die funktionierende LSTM-Version liegt zur Dokumentation auf dem Branch `submission-with-lstm`, ist aber nicht die finale Abgabe.
