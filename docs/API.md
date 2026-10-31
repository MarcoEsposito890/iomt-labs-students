# La libreria del corso

`iomtlab` contiene il codice che i laboratori usano: lettura dei CSV, tempi e
finestre, feature, split per soggetto, metriche, modelli, addestramento,
compressione, inferenza distribuita, incertezza e apprendimento federato. Tutte
le funzioni sono complete e coperte da test in `tests/`.

I notebook importano la libreria per mostrare i risultati. Quando una funzione
è l'argomento del laboratorio, il notebook ne mostra il sorgente per intero,
copiato da `src/`, e il docente lo commenta.

| modulo | contenuto |
|---|---|
| `net` | framing su TCP, formato del payload, pacchetti MQTT, tap e misure del Lab 1 |
| `signal` | lettura delle prove, tempi di consegna, modello dell'orologio e accordo delle etichette (`shift_clock`, `label_agreement`, `offset_after`), ricampionamento, finestre; `build_windows` le rifà tutte dal Lab 3 |
| `tug` | fasi e durate del Timed Up and Go, incertezza di quantizzazione |
| `features` | feature nel dominio del tempo, a blocchi |
| `io` | file che passano da un laboratorio al successivo, fra cui `save_windows` (`work/windows.npz`) |
| `evaluation` | pieghe per soggetto, metriche, tabelle |
| `evaluation_figures` | validazioni a confronto, matrice di confusione, loss per epoca, confidenza |
| `forests` | validazione incrociata con un modello nuovo per piega, costo di una foresta e delle feature sul nodo |
| `train` | ciclo di addestramento, semi, pesi di classe; finestre e pieghe di riferimento dei Lab 4–8, le quattro reti addestrate con una ricetta (`RECIPE`, `fit_reference`, `reference_model`, `train_windows`) |
| `models` | SimpleCNN, ExampleCNN (la rete di esempio della teoria), BottleneckCNN, DeepConvLSTM e il loro costo; `models.memory` la RAM di un'inferenza; `models.contract.check_model` e `train.register_model` per il modello di un progetto; `uncounted_modules` per gli strati non contati |
| `edge` | misure sul modello, taglio fra nodo e gateway con h salti radio (`cut_cost`), int8; `edge.cuts` i tagli di una rete con `stages()`, il loro ritardo, il taglio vero con `pack_int8`, un'uscita al taglio e la sua curva, `edge.split_figures` le loro figure (Lab 6); `edge.fusion` fusione dei punteggi di più nodi, allineamento per evento, costo del consenso e del nodo che decide (Lab 8, progetto); `edge.regimes` dove vince ogni taglio: la regola m + z < 1, l'inviluppo lungo k = ρh, collegamenti con trame e LoRa, vincoli e scelta del più economico fra i fattibili (Lab 6, tesine); `edge.compress` quantizzazione, pruning, distillazione e i quattro test sul nodo, `edge.compress_figures` le loro figure |
| `radio` | trame IEEE 802.15.4: 116 B utili, 17 B di intestazioni, tempo ed energia sull'aria |
| `uncertainty` | calibrazione, insiemi conformi; instradamento sui singoletti con energia e nodi per canale (`routing_table`, `energy_constants`, `link_load`); `uncertainty.figures` le figure del Lab 7 |
| `fl` | federated averaging, partizioni dei client; costo di un round su 802.15.4, FedAvg, centralizzato e SGD decentralizzato con la storia dei round (`fl.rounds`); grafi, pesi di Metropolis, autovalore e consenso (`fl.consensus`); `fl.figures` le figure dei Lab 8 e 9 |
| `plots` | le figure ricorrenti; quelle del Lab 2 (`plot_phases`, `plot_intervals`, `plot_agreement`, `plot_window_purity`) |
| `config` | costanti della pipeline e dell'energia |
| `spec` | `DatasetSpec`, la descrizione di un dataset; `use`, `active`, `using`: il dataset su cui lavora la libreria (default TUG); vedi `projects/README.md` |

`help(funzione)` stampa firma e documentazione. Le operazioni da una riga di
numpy, pandas, scipy, sklearn o matplotlib non hanno una funzione dedicata.

---

## Indice per laboratorio

*Generato dal codice e dai notebook.*

| Lab | notebook | usa dalla libreria |
|---|---|---|
| 0 | lab00_setup.ipynb | `net.broker_available`, `net.mqtt_client`, `plots.plot_trial`, `signal.list_executions`, `signal.load_execution` |
| 1 | lab01_mqtt.ipynb | `features.extract`, `net.broker_available`, `net.encode_json`, `net.encoding_sizes`, `net.fanout_table`, `net.framed_transfer`, `net.mqtt_cost`, `net.overhead_table`, `net.pack_window`, `net.path_table`, `net.plot_encodings`, `net.plot_overhead`, `net.plot_paths`, `net.plot_recv`, `net.publish_size`, `net.recv_sizes`, `net.synthetic_window`, `net.unpack_window` |
| 2 | lab02_dataset.ipynb | `features.extract`, `io.save_features`, `io.save_windows`, `plots.plot_agreement`, `plots.plot_intervals`, `plots.plot_phases`, `plots.plot_trial`, `plots.plot_window_purity`, `signal.align_to_common_grid`, `signal.build_windows`, `signal.device_offset`, `signal.list_executions`, `signal.load_execution`, `signal.load_executions_info`, `signal.load_subjects_info`, `signal.offset_after`, `signal.resample_to_hz`, `signal.timing_report` |
| 3 | lab03_method_forests.ipynb | `evaluation.confusion_report`, `evaluation.kfold_shuffled`, `evaluation.loso_folds`, `evaluation.macro_f1`, `evaluation.per_subject_scores`, `evaluation.subject_folds`, `evaluation.top_k_table`, `evaluation_figures.plot_confusion`, `evaluation_figures.plot_validation`, `features.extract`, `forests.evaluate_forest`, `forests.features_cost`, `forests.mdi_orders`, `io.fold_masks`, `io.load_features`, `io.save_features`, `plots.per_subject_bars`, `plots.plot_cost_accuracy`, `plots.plot_folds`, `signal.build_windows`, `train.save_reference_split` |
| 4 | lab04_four_cnns.ipynb | `edge.time_per_window`, `evaluation.confusion_report`, `evaluation.per_subject_scores`, `evaluation_figures.plot_confidence`, `evaluation_figures.plot_confusion`, `evaluation_figures.plot_losses`, `models.BottleneckCNN`, `models.ExampleCNN`, `models.count_parameters`, `models.layer_profile`, `models.model_size_bytes`, `models.memory.peak_activation_bytes`, `plots.plot_cost_accuracy`, `signal.normalize_full_scale`, `train.fit_reference`, `train.load_reference_windows`, `train.reference_split`, `train.train_test_masks` |
| 5 | lab05_compression.ipynb | `edge.nonzero_params`, `edge.size_on_disk`, `edge.compress.dequantize_state`, `edge.compress.int8_bytes`, `edge.compress.peak_activation_values`, `edge.compress.prune_magnitude`, `edge.compress.quantize_state`, `edge.compress.shrink_example_cnn`, `edge.compress.state_bytes`, `edge.compress.train_student`, `edge.compress.update_on_link`, `edge.compress.weight_error`, `edge.compress_figures.plot_bits_f1`, `edge.compress_figures.plot_distillation`, `edge.compress_figures.plot_four_tests`, `edge.compress_figures.plot_model_bytes`, `edge.compress_figures.plot_pruning`, `evaluation.macro_f1`, `models.ExampleCNN`, `models.count_parameters`, `models.layer_profile`, `radio.air_bytes`, `radio.frames`, `train.load_reference_windows`, `train.logits_of`, `train.reference_model`, `train.train_test_masks` |
| 6 | lab06_split_inference.ipynb | `edge.pack_int8`, `edge.split_model`, `edge.unpack_int8`, `edge.cuts.cut_delay`, `edge.cuts.cut_profile`, `edge.cuts.train_exit`, `edge.regimes.envelope`, `edge.regimes.rule`, `edge.split_figures.plot_bottleneck_sweep`, `edge.split_figures.plot_cut_bits`, `edge.split_figures.plot_cut_energy`, `edge.split_figures.plot_exit_curve`, `edge.split_figures.plot_mz`, `edge.split_figures.plot_regimes`, `evaluation.macro_f1`, `io.save_curve`, `models.BottleneckCNN`, `net.broker_available`, `net.mqtt_cost`, `net.publish_size`, `radio.air_bytes`, `radio.air_energy_J`, `radio.air_time_s`, `radio.frames`, `train.load_reference_windows`, `train.logits_of`, `train.reference_model`, `train.set_seed`, `train.to_input`, `train.train_test_masks`, `train.train_windows` |
| 7 | lab07_conformal_routing.ipynb | `evaluation.macro_f1`, `io.load_curve`, `models.ExampleCNN`, `plots.reliability_diagram`, `train.load_reference_windows`, `train.logits_of`, `train.reference_model`, `train.reference_split`, `train.set_seed`, `train.train_windows`, `uncertainty.apply_temperature`, `uncertainty.coverage_by_class`, `uncertainty.coverage_by_subject`, `uncertainty.ece`, `uncertainty.energy_constants`, `uncertainty.energy_per_decision`, `uncertainty.fit_temperature`, `uncertainty.link_load`, `uncertainty.nonconformity`, `uncertainty.routing_table`, `uncertainty.singleton_rate`, `uncertainty.figures.plot_calibration_pair`, `uncertainty.figures.plot_coverage`, `uncertainty.figures.plot_routing` |
| 8 | lab08_fusion_consensus.ipynb | `edge.cuts.cut_profile`, `edge.fusion.consensus_cost`, `edge.fusion.fuse`, `edge.fusion.fuse_confidence`, `edge.fusion.fuse_mean`, `edge.fusion.gateway_cost`, `edge.fusion.score_bits`, `edge.fusion.softmax`, `edge.regimes.LINKS`, `edge.regimes.link_cost`, `evaluation.macro_f1`, `fl.disagreement`, `fl.grid_graph`, `fl.iterations_to`, `fl.metropolis_weights`, `fl.ring_graph`, `fl.run_consensus`, `fl.second_eigenvalue`, `fl.figures.plot_consensus`, `models.ExampleCNN`, `signal.build_paired_windows`, `train.logits_of`, `train.make_loader`, `train.set_seed`, `train.to_input`, `train.train_test_masks` |
| 9 | lab09_federated.ipynb (facoltativo) | `fl.bytes_to_reach`, `fl.dirichlet_partition`, `fl.disagreement`, `fl.grid_graph`, `fl.iterations_to`, `fl.ring_graph`, `fl.round_cost`, `fl.run_centralised`, `fl.run_consensus`, `fl.run_dsgd`, `fl.run_fedavg`, `fl.second_eigenvalue`, `fl.subject_clients`, `fl.uniform_weights`, `fl.update_bytes`, `fl.figures.plot_client_mix`, `fl.figures.plot_consensus`, `fl.figures.plot_round_cost`, `models.ExampleCNN`, `models.count_parameters`, `models.federated_round_bytes`, `plots.plot_rounds`, `radio.air_bytes`, `radio.air_energy_J`, `radio.air_time_s`, `radio.frames`, `train.load_reference_windows`, `train.set_seed`, `train.train_test_masks` |

---

## Metriche

La macro-F1 del corso è la media delle F1 di tutte e cinque le classi.
Se in una partizione una classe non compare né nelle etichette né nelle
predizioni, la sua F1 vale 0 e resta nella media.

```python
from sklearn.metrics import f1_score
f1_score(y_true, y_pred, labels=list(range(5)), average="macro", zero_division=0)
```

Senza `labels` sklearn fa la media solo sulle classi presenti, e il numero
cambia fra partizioni. È la stessa definizione di `evaluation.macro_f1` e di
`train.evaluate`.

---

## Convenzioni fissate

Sono decisioni, non proprietà del segnale: due implementazioni corrette le
potrebbero risolvere diversamente, e i test in `tests/` le controllano.

| funzione | convenzione |
|---|---|
| `timing_report` | deviazione standard di popolazione (`ddof=0`); i timestamp duplicati restano come intervalli nulli; un buco è Δt > 3 × mediana |
| `resample_to_hz` | griglia da 0 s, nessun istante oltre l'ultimo campione; etichetta dell'ultimo campione grezzo non successivo all'istante |
| `rfft_magnitude` | media rimossa prima della finestra; ampiezza a singolo lato corretta per il guadagno coerente |
| `make_windows`, `label_window` | a parità di conteggio vince l'etichetta con indice minore; `label_window` dichiara pura una finestra con almeno l'80 % di una classe |
| `align_to_common_grid` | la griglia copre l'intersezione dei due intervalli; nessuna estrapolazione |
| `lowpass_ma` | media mobile applicata in avanti e all'indietro: fase nulla |
