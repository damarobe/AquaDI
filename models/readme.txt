README – CPPN generavimo įrankiai
=================================

Šiame kataloge rasite du Python skriptus, skirtus kurti spalvotus, mandalų bei organinių raštų generavimui naudojant CPPN (Compositional Pattern Producing Network) principą.

1. cppn_generator.py  
   – Generuoja vieną CPPN atvaizdą pagal vartotojo nurodytus parametrus.  
   – Pagrindiniai parametrai:  
     • ––width, ––height    – paveikslo raiška (pvz. 512×512)  
     • ––latent             – latentinio vektoriaus matmuo  
     • ––depth              – paslėptų sluoksnių skaičius  
     • ––width_h            – sluoksnių pločio (neuronių skaičiaus) parametras  
     • ––var                – VarianceScaling inicializatoriaus skalė  
     • ––warp_alpha         – koordinatų „iškraipymo“ stiprumas  
     • ––seed               – atsitiktinio generatoriaus sėkla  
     • ––out                – sugeneruoto PNG failo pavadinimas  

   Pvz.:  
     python cppn_generator.py --width 512 --height 512 --latent 2 --depth 6 --width_h 128 --var 100 --warp_alpha 0.8 --seed 42 --out my_pattern.png

2. batched_cppn_generator.py  
   – Leidžia vienu paleidimu sukurti daug CPPN atvaizdų su skirtingais hiperparametrais.  
   – Vartotojas nurodo:  
     • ––out_dir            – išvesties katalogas, kur bus saugomi PNG  
     • ––count              – sugeneruojamų paveikslų skaičius  
     • ––seed               – pradinė sėkla; vėliau kiekvienam atvaizdui pridedama +1  
     • ––var_min/––var_max  – „variance“ intervalo reikšmės (nuo–iki)  
     • ––warp_min/––warp_max– koordinatų iškraipymo stiprumo intervalas  
     • ––freq_min/––freq_max– koordinatų warp dažnio intervalas  
     • ––depth_min/––depth_max – sluoksnių skaičiaus intervalas  
     • ––width_min/––width_max – sluoksnio pločio (neuronių) intervalas  
     • ––latent_dim         – latentinio vektoriaus matmuo  

   Pvz.:  
     python batched_cppn_generator.py \
       --out_dir results/diverse_cppn \
       --count 20 \
       --width 512 --height 512 \
       --latent_dim 2 \
       --seed 42 \
       --var_min 50 --var_max 300 \
       --warp_min 0.1 --warp_max 1.5 \
       --freq_min 1.0 --freq_max 10.0 \
       --depth_min 4 --depth_max 8 \
       --width_min 64 --width_max 256

Priklausomybės
--------------
- Python 3.7+  
- numpy  
- matplotlib  
- scikit-image  
- pillow  

Diegimas
---------
1. Sukurkite virtualią aplinką (rekomenduojama):  
     python -m venv venv  
     source venv/bin/activate  (Linux/Mac)  
     venv\Scripts\activate     (Windows)  
2. Įdiekite priklausomybes:  
     pip install numpy matplotlib scikit-image pillow  

Naudojimas
----------
- Paleiskite `cppn_generator.py`, jei norite greitai sugeneruoti vieną vaizdą su specifiniais parametrais.  
- Paleiskite `batched_cppn_generator.py`, kad sukurtumėte didelį serijų rinkinį su varijuojančiais parametrais.  

