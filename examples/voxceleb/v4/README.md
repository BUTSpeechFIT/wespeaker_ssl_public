

## Change Log

🔥 UPDATE 2024.08.31: Now support MHFA [1] on top of WavLM features, as well as WavLM models pre-trained on the VoxCeleb2-dev dataset [2]. MHFA-WavLM significantly shortens the training epochs to 23 while achieving strong performance.

## Todo

- [ ] Support Parameter-efficient tuning (e.g. adapter, prefix tuning)
- [ ] Support ECAPA_TDNN model pre-trained with Hubert loss on the VoxCeleb2 dataset.

## Results

* Setup: raw waveform, epoch25, ArcMargin, aug_prob0.6
* Scoring: cosine (sub mean of vox2_dev), AS-Norm, [QMF](https://arxiv.org/pdf/2010.11255)
* Metric: EER(%)

## WavLM results

* Pre-trained frontend: the [WavLM](https://arxiv.org/abs/2110.13900) Large model, multilayer features are used
* Speaker model: MHFA
* Training strategy: Frozen => Joint ft => Joint lmft

```bash
bash run_MHFA_Base.sh --stage 3 --stop_stage 9

bash run_MHFA_Large.sh --stage 3 --stop_stage 9
```

| Model | AS-Norm | LMFT | QMF | vox1-O-clean | vox1-E-clean | vox1-H-clean |
|:------------------|:-------:|:---|:---:|:------------:|:------------:|:------------:|
| WavLM Base Plus + MHFA            | √ | × | × | 0.750 | 0.716 | 1.442 |
| WavLM Large + MHFA            | √ | × | × | 0.649 | 0.610 | 1.235 |


```

```

## Citation

```bibtex
@INPROCEEDINGS{10022775,
  author={Peng, Junyi and Plchot, Oldřich and Stafylakis, Themos and Mošner, Ladislav and Burget, Lukáš and Černocký, Jan},
  booktitle={2022 IEEE Spoken Language Technology Workshop (SLT)}, 
  title={An Attention-Based Backend Allowing Efficient Fine-Tuning of Transformer Models for Speaker Verification}, 
  year={2023},
  volume={},
  number={},
  pages={555-562},
```

```****

```
