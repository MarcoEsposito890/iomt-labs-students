# Laboratori IoT/IoMT — dal sensore al federated learning

Nove laboratori su un unico filo conduttore: riconoscere le fasi di un test
Timed Up and Go da segnali inerziali, dal sensore grezzo fino a un sistema
federato. Un laboratorio è la teoria vista in pratica: il notebook gira
dall'inizio alla fine, il docente lo commenta, e si cambiano i parametri
indicati per vedere che cosa succede. Ogni laboratorio chiude su un numero
misurato, con la sua fonte.

| Lab | lezione | argomento |
|---|---|---|
| 0 | 11 | l'ambiente di lavoro: Python, Git e il repository, broker MQTT |
| 1 | 13 | TCP grezzo e MQTT: framing, formato del payload, byte per messaggio e per QoS |
| 2 | 15 | dal segnale al vettore di feature |
| 3 | 18 | metodo sperimentale, alberi e foreste |
| 4 | 21 | tre CNN: costo e prestazione |
| 5 | 24 | compressione: int8, pruning, distillazione |
| 6 | 26 | inferenza distribuita: dove tagliare |
| 7 | 29 | calibrazione, insiemi conformi, instradamento |
| 8 | da fissare | più nodi sullo stesso evento: fusione e consenso |
| 9 | 31 | federated averaging, e senza server |

Il Lab 2 produce il vettore di feature che il Lab 3 consuma; `labs/README.md`
dice quali notebook sono pronti.

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install torch --index-url https://download.pytorch.org/whl/cpu   # prima: evita i wheel CUDA
pip install -e ".[dev,net]"
python scripts/get_data.py        # dataset da Zenodo: serve dal Lab 2 (i Lab 0 e 1 non lo usano)
python scripts/verify_setup.py --quick
```

I Lab 0 e 1 richiedono un broker MQTT su `localhost:1883`: si installa
[Mosquitto](https://mosquitto.org/download/) (macOS: `brew install mosquitto`;
Linux: il pacchetto `mosquitto`) e lo si avvia con `mosquitto -v`.
`verify_setup.py` dice se risponde.

Si lavora in `work/`, non in `labs/`. A ogni lezione:

```bash
git pull
cp labs/lab01_mqtt.ipynb work/
jupyter lab work/lab01_mqtt.ipynb
```

`work/` è ignorata da git, quindi `git pull` non entra mai in conflitto con il
proprio lavoro. La libreria `iomtlab` è completa: i notebook la importano, e
nel progetto finale la si estende. `docs/API.md` dice che cosa usa ogni
laboratorio.

## Dati

Matey-Sanz, Casteleyn, Granell, *Data in Brief* 51, 2023.
23 soggetti, 223 acquisizioni valide per dispositivo, 5 classi.
CC-BY 4.0, DOI `10.5281/zenodo.8398688`.
