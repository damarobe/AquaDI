# Aqua12345: Dirbtinio intelekto pagrindu veikianti garso-vaizdo sintezės sistema

**Aqua12345** – tai meninės raiškos sistema, kuri naudoja dirbtinį intelektą ir skysčių dinamikos modeliavimą, kad pagal garso signalus generuotų vizualiai patrauklius paveikslus. Projektas jungia balso analizę, fizikinį modeliavimą ir generatyvinio meno metodus.

---

## 🔧 Projekto struktūra

AquaDI/

├── data/ # Duomenys: garso įrašai, paveikslai, sugeneruoti rezultatai

├── evaluation/ # Metrikų (LPIPS, SSIM, PSNR) skaičiavimas ir vaizdų analizė

├── sample images/ # Pavyzdiniai paveikslai sugeneruoti naudojant įvairius DI modelius

├── scripts/ # Vaizdo generavimo DI  modeliai

├── requirements.txt # Reikalingos bibliotekos

└── README.md # Šis failas


---

## 🚀 Paleidimas

### 1. Sukurkite virtualią aplinką (pasirinktinai)

python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

### 2. Įdiekite reikiamas priklausomybes

pip install -r requirements.txt

### 3. Sugeneruokite paveikslus

python models/cppn_generator.py

### 4. Įvertinkite kokybę (LPIPS, SSIM, PSNR)

python evaluation/metrics.py --input_dir data/generated_outputs/

## 📜 Licencija
Projektas pateikiamas pagal CC0-1.0 license licenciją. Daugiau informacijos – LICENSE faile.

## 🤝 Indėlis
Norite prisidėti? Pateikite „pull request“, pasiūlykite idėjų ar užregistruokite „issue“. 
Atviras mokslas – geresnis mokslas!
