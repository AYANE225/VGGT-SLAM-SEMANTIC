# VGGT-SEM 项目文档

## 项目概述

VGGT-SEM是一个视觉-几何-语义融合的SLAM系统。

## 项目结构

```
.
|-- app.py
|-- backups
|   `-- vggt_slam_20260116_020631.tar.gz
|-- DATA
|   |-- corridor_0300-0399
|   |   |-- frame_0300.jpg
|   |   |-- frame_0301.jpg
|   |   |-- frame_0302.jpg
|   |   |-- frame_0303.jpg
|   |   |-- frame_0304.jpg
|   |   |-- frame_0305.jpg
|   |   |-- frame_0306.jpg
|   |   |-- frame_0307.jpg
|   |   |-- frame_0308.jpg
|   |   |-- frame_0309.jpg
|   |   |-- frame_0310.jpg
|   |   |-- frame_0311.jpg
|   |   |-- frame_0312.jpg
|   |   |-- frame_0313.jpg
|   |   |-- frame_0314.jpg
|   |   |-- frame_0315.jpg
|   |   |-- frame_0316.jpg
|   |   |-- frame_0317.jpg
|   |   |-- frame_0318.jpg
|   |   |-- frame_0319.jpg
|   |   |-- frame_0320.jpg
|   |   |-- frame_0321.jpg
|   |   |-- frame_0322.jpg
|   |   |-- frame_0323.jpg
|   |   |-- frame_0324.jpg
|   |   |-- frame_0325.jpg
|   |   |-- frame_0326.jpg
|   |   |-- frame_0327.jpg
|   |   |-- frame_0328.jpg
|   |   |-- frame_0329.jpg
|   |   |-- frame_0330.jpg
|   |   |-- frame_0331.jpg
|   |   |-- frame_0332.jpg
|   |   |-- frame_0333.jpg
|   |   |-- frame_0334.jpg
|   |   |-- frame_0335.jpg
|   |   |-- frame_0336.jpg
|   |   |-- frame_0337.jpg
|   |   |-- frame_0338.jpg
|   |   |-- frame_0339.jpg
|   |   |-- frame_0340.jpg
|   |   |-- frame_0341.jpg
|   |   |-- frame_0342.jpg
|   |   |-- frame_0343.jpg
|   |   |-- frame_0344.jpg
|   |   |-- frame_0345.jpg
|   |   |-- frame_0346.jpg
|   |   |-- frame_0347.jpg
|   |   |-- frame_0348.jpg
|   |   |-- frame_0349.jpg
|   |   |-- frame_0350.jpg
|   |   |-- frame_0351.jpg
|   |   |-- frame_0352.jpg
|   |   |-- frame_0353.jpg
|   |   |-- frame_0354.jpg
|   |   |-- frame_0355.jpg
|   |   |-- frame_0356.jpg
|   |   |-- frame_0357.jpg
|   |   |-- frame_0358.jpg
|   |   |-- frame_0359.jpg
|   |   |-- frame_0360.jpg
|   |   |-- frame_0361.jpg
|   |   |-- frame_0362.jpg
|   |   |-- frame_0363.jpg
|   |   |-- frame_0364.jpg
|   |   |-- frame_0365.jpg
|   |   |-- frame_0366.jpg
|   |   |-- frame_0367.jpg
|   |   |-- frame_0368.jpg
|   |   |-- frame_0369.jpg
|   |   |-- frame_0370.jpg
|   |   |-- frame_0371.jpg
|   |   |-- frame_0372.jpg
|   |   |-- frame_0373.jpg
|   |   |-- frame_0374.jpg
|   |   |-- frame_0375.jpg
|   |   |-- frame_0376.jpg
|   |   |-- frame_0377.jpg
|   |   |-- frame_0378.jpg
|   |   |-- frame_0379.jpg
|   |   |-- frame_0380.jpg
|   |   |-- frame_0381.jpg
|   |   |-- frame_0382.jpg
|   |   |-- frame_0383.jpg
|   |   |-- frame_0384.jpg
|   |   |-- frame_0385.jpg
|   |   |-- frame_0386.jpg
|   |   |-- frame_0387.jpg
|   |   |-- frame_0388.jpg
|   |   |-- frame_0389.jpg
|   |   |-- frame_0390.jpg
|   |   |-- frame_0391.jpg
|   |   |-- frame_0392.jpg
|   |   |-- frame_0393.jpg
|   |   |-- frame_0394.jpg
|   |   |-- frame_0395.jpg
|   |   |-- frame_0396.jpg
|   |   |-- frame_0397.jpg
|   |   |-- frame_0398.jpg
|   |   `-- frame_0399.jpg
|   `-- office_loop
|       |-- frame_0001.jpg
|       |-- frame_0002.jpg
|       |-- frame_0003.jpg
|       |-- frame_0004.jpg
|       |-- frame_0005.jpg
|       |-- frame_0006.jpg
|       |-- frame_0007.jpg
|       |-- frame_0008.jpg
|       |-- frame_0009.jpg
|       |-- frame_0010.jpg
|       |-- frame_0011.jpg
|       |-- frame_0012.jpg
|       |-- frame_0013.jpg
|       |-- frame_0014.jpg
|       |-- frame_0015.jpg
|       |-- frame_0016.jpg
|       |-- frame_0017.jpg
|       |-- frame_0018.jpg
|       |-- frame_0019.jpg
|       |-- frame_0020.jpg
|       |-- frame_0021.jpg
|       |-- frame_0022.jpg
|       |-- frame_0023.jpg
|       |-- frame_0024.jpg
|       |-- frame_0025.jpg
|       |-- frame_0026.jpg
|       |-- frame_0027.jpg
|       |-- frame_0028.jpg
|       |-- frame_0029.jpg
|       |-- frame_0030.jpg
|       |-- frame_0031.jpg
|       |-- frame_0032.jpg
|       |-- frame_0033.jpg
|       |-- frame_0034.jpg
|       |-- frame_0035.jpg
|       |-- frame_0036.jpg
|       |-- frame_0037.jpg
|       |-- frame_0038.jpg
|       |-- frame_0039.jpg
|       |-- frame_0040.jpg
|       |-- frame_0041.jpg
|       |-- frame_0042.jpg
|       |-- frame_0043.jpg
|       |-- frame_0044.jpg
|       |-- frame_0045.jpg
|       |-- frame_0046.jpg
|       |-- frame_0047.jpg
|       |-- frame_0048.jpg
|       |-- frame_0049.jpg
|       |-- frame_0050.jpg
|       |-- frame_0051.jpg
|       |-- frame_0052.jpg
|       |-- frame_0053.jpg
|       |-- frame_0054.jpg
|       |-- frame_0055.jpg
|       |-- frame_0056.jpg
|       |-- frame_0057.jpg
|       |-- frame_0058.jpg
|       |-- frame_0059.jpg
|       |-- frame_0060.jpg
|       |-- frame_0061.jpg
|       |-- frame_0062.jpg
|       |-- frame_0063.jpg
|       |-- frame_0064.jpg
|       |-- frame_0065.jpg
|       |-- frame_0066.jpg
|       |-- frame_0067.jpg
|       |-- frame_0068.jpg
|       |-- frame_0069.jpg
|       |-- frame_0070.jpg
|       |-- frame_0071.jpg
|       |-- frame_0072.jpg
|       |-- frame_0073.jpg
|       |-- frame_0074.jpg
|       |-- frame_0075.jpg
|       |-- frame_0076.jpg
|       |-- frame_0077.jpg
|       |-- frame_0078.jpg
|       |-- frame_0079.jpg
|       |-- frame_0080.jpg
|       |-- frame_0081.jpg
|       |-- frame_0082.jpg
|       |-- frame_0083.jpg
|       |-- frame_0084.jpg
|       |-- frame_0085.jpg
|       |-- frame_0086.jpg
|       |-- frame_0087.jpg
|       |-- frame_0088.jpg
|       |-- frame_0089.jpg
|       |-- frame_0090.jpg
|       |-- frame_0091.jpg
|       |-- frame_0092.jpg
|       |-- frame_0093.jpg
|       |-- frame_0094.jpg
|       |-- frame_0095.jpg
|       |-- frame_0096.jpg
|       |-- frame_0097.jpg
|       |-- frame_0098.jpg
|       |-- frame_0099.jpg
|       |-- frame_0100.jpg
|       |-- frame_0101.jpg
|       |-- frame_0102.jpg
|       |-- frame_0103.jpg
|       |-- frame_0104.jpg
|       |-- frame_0105.jpg
|       |-- frame_0106.jpg
|       |-- frame_0107.jpg
|       |-- frame_0108.jpg
|       |-- frame_0109.jpg
|       |-- frame_0110.jpg
|       |-- frame_0111.jpg
|       |-- frame_0112.jpg
|       |-- frame_0113.jpg
|       |-- frame_0114.jpg
|       |-- frame_0115.jpg
|       |-- frame_0116.jpg
|       |-- frame_0117.jpg
|       |-- frame_0118.jpg
|       |-- frame_0119.jpg
|       |-- frame_0120.jpg
|       |-- frame_0121.jpg
|       |-- frame_0122.jpg
|       |-- frame_0123.jpg
|       |-- frame_0124.jpg
|       |-- frame_0125.jpg
|       |-- frame_0126.jpg
|       |-- frame_0127.jpg
|       |-- frame_0128.jpg
|       |-- frame_0129.jpg
|       |-- frame_0130.jpg
|       |-- frame_0131.jpg
|       |-- frame_0132.jpg
|       |-- frame_0133.jpg
|       |-- frame_0134.jpg
|       |-- frame_0135.jpg
|       |-- frame_0136.jpg
|       |-- frame_0137.jpg
|       |-- frame_0138.jpg
|       |-- frame_0139.jpg
|       |-- frame_0140.jpg
|       |-- frame_0141.jpg
|       |-- frame_0142.jpg
|       |-- frame_0143.jpg
|       |-- frame_0144.jpg
|       |-- frame_0145.jpg
|       |-- frame_0146.jpg
|       |-- frame_0147.jpg
|       |-- frame_0148.jpg
|       |-- frame_0149.jpg
|       |-- frame_0150.jpg
|       |-- frame_0151.jpg
|       |-- frame_0152.jpg
|       |-- frame_0153.jpg
|       |-- frame_0154.jpg
|       |-- frame_0155.jpg
|       |-- frame_0156.jpg
|       |-- frame_0157.jpg
|       |-- frame_0158.jpg
|       |-- frame_0159.jpg
|       |-- frame_0160.jpg
|       |-- frame_0161.jpg
|       |-- frame_0162.jpg
|       |-- frame_0163.jpg
|       |-- frame_0164.jpg
|       |-- frame_0165.jpg
|       |-- frame_0166.jpg
|       |-- frame_0167.jpg
|       |-- frame_0168.jpg
|       |-- frame_0169.jpg
|       |-- frame_0170.jpg
|       |-- frame_0171.jpg
|       |-- frame_0172.jpg
|       |-- frame_0173.jpg
|       |-- frame_0174.jpg
|       |-- frame_0175.jpg
|       |-- frame_0176.jpg
|       |-- frame_0177.jpg
|       |-- frame_0178.jpg
|       |-- frame_0179.jpg
|       |-- frame_0180.jpg
|       |-- frame_0181.jpg
|       |-- frame_0182.jpg
|       |-- frame_0183.jpg
|       |-- frame_0184.jpg
|       |-- frame_0185.jpg
|       |-- frame_0186.jpg
|       |-- frame_0187.jpg
|       |-- frame_0188.jpg
|       |-- frame_0189.jpg
|       |-- frame_0190.jpg
|       |-- frame_0191.jpg
|       |-- frame_0192.jpg
|       |-- frame_0193.jpg
|       |-- frame_0194.jpg
|       |-- frame_0195.jpg
|       |-- frame_0196.jpg
|       |-- frame_0197.jpg
|       |-- frame_0198.jpg
|       |-- frame_0199.jpg
|       |-- frame_0200.jpg
|       |-- frame_0201.jpg
|       |-- frame_0202.jpg
|       |-- frame_0203.jpg
|       |-- frame_0204.jpg
|       |-- frame_0205.jpg
|       |-- frame_0206.jpg
|       |-- frame_0207.jpg
|       |-- frame_0208.jpg
|       |-- frame_0209.jpg
|       |-- frame_0210.jpg
|       |-- frame_0211.jpg
|       |-- frame_0212.jpg
|       |-- frame_0213.jpg
|       |-- frame_0214.jpg
|       |-- frame_0215.jpg
|       |-- frame_0216.jpg
|       |-- frame_0217.jpg
|       |-- frame_0218.jpg
|       |-- frame_0219.jpg
|       |-- frame_0220.jpg
|       |-- frame_0221.jpg
|       |-- frame_0222.jpg
|       |-- frame_0223.jpg
|       |-- frame_0224.jpg
|       |-- frame_0225.jpg
|       |-- frame_0226.jpg
|       |-- frame_0227.jpg
|       |-- frame_0228.jpg
|       |-- frame_0229.jpg
|       |-- frame_0230.jpg
|       |-- frame_0231.jpg
|       |-- frame_0232.jpg
|       |-- frame_0233.jpg
|       |-- frame_0234.jpg
|       |-- frame_0235.jpg
|       |-- frame_0236.jpg
|       |-- frame_0237.jpg
|       |-- frame_0238.jpg
|       |-- frame_0239.jpg
|       |-- frame_0240.jpg
|       |-- frame_0241.jpg
|       |-- frame_0242.jpg
|       |-- frame_0243.jpg
|       |-- frame_0244.jpg
|       |-- frame_0245.jpg
|       |-- frame_0246.jpg
|       |-- frame_0247.jpg
|       |-- frame_0248.jpg
|       |-- frame_0249.jpg
|       |-- frame_0250.jpg
|       |-- frame_0251.jpg
|       |-- frame_0252.jpg
|       |-- frame_0253.jpg
|       |-- frame_0254.jpg
|       |-- frame_0255.jpg
|       |-- frame_0256.jpg
|       |-- frame_0257.jpg
|       |-- frame_0258.jpg
|       |-- frame_0259.jpg
|       |-- frame_0260.jpg
|       |-- frame_0261.jpg
|       |-- frame_0262.jpg
|       |-- frame_0263.jpg
|       |-- frame_0264.jpg
|       |-- frame_0265.jpg
|       |-- frame_0266.jpg
|       |-- frame_0267.jpg
|       |-- frame_0268.jpg
|       |-- frame_0269.jpg
|       |-- frame_0270.jpg
|       |-- frame_0271.jpg
|       |-- frame_0272.jpg
|       |-- frame_0273.jpg
|       |-- frame_0274.jpg
|       |-- frame_0275.jpg
|       |-- frame_0276.jpg
|       |-- frame_0277.jpg
|       |-- frame_0278.jpg
|       |-- frame_0279.jpg
|       |-- frame_0280.jpg
|       |-- frame_0281.jpg
|       |-- frame_0282.jpg
|       |-- frame_0283.jpg
|       |-- frame_0284.jpg
|       |-- frame_0285.jpg
|       |-- frame_0286.jpg
|       |-- frame_0287.jpg
|       |-- frame_0288.jpg
|       |-- frame_0289.jpg
|       |-- frame_0290.jpg
|       |-- frame_0291.jpg
|       |-- frame_0292.jpg
|       |-- frame_0293.jpg
|       |-- frame_0294.jpg
|       |-- frame_0295.jpg
|       |-- frame_0296.jpg
|       |-- frame_0297.jpg
|       |-- frame_0298.jpg
|       |-- frame_0299.jpg
|       |-- frame_0300.jpg
|       |-- frame_0301.jpg
|       |-- frame_0302.jpg
|       |-- frame_0303.jpg
|       |-- frame_0304.jpg
|       |-- frame_0305.jpg
|       |-- frame_0306.jpg
|       |-- frame_0307.jpg
|       |-- frame_0308.jpg
|       |-- frame_0309.jpg
|       |-- frame_0310.jpg
|       |-- frame_0311.jpg
|       |-- frame_0312.jpg
|       |-- frame_0313.jpg
|       |-- frame_0314.jpg
|       |-- frame_0315.jpg
|       |-- frame_0316.jpg
|       |-- frame_0317.jpg
|       |-- frame_0318.jpg
|       |-- frame_0319.jpg
|       |-- frame_0320.jpg
|       |-- frame_0321.jpg
|       |-- frame_0322.jpg
|       |-- frame_0323.jpg
|       |-- frame_0324.jpg
|       |-- frame_0325.jpg
|       |-- frame_0326.jpg
|       |-- frame_0327.jpg
|       |-- frame_0328.jpg
|       |-- frame_0329.jpg
|       |-- frame_0330.jpg
|       |-- frame_0331.jpg
|       |-- frame_0332.jpg
|       |-- frame_0333.jpg
|       |-- frame_0334.jpg
|       |-- frame_0335.jpg
|       |-- frame_0336.jpg
|       |-- frame_0337.jpg
|       |-- frame_0338.jpg
|       |-- frame_0339.jpg
|       |-- frame_0340.jpg
|       |-- frame_0341.jpg
|       |-- frame_0342.jpg
|       |-- frame_0343.jpg
|       |-- frame_0344.jpg
|       |-- frame_0345.jpg
|       |-- frame_0346.jpg
|       |-- frame_0347.jpg
|       |-- frame_0348.jpg
|       |-- frame_0349.jpg
|       |-- frame_0350.jpg
|       |-- frame_0351.jpg
|       |-- frame_0352.jpg
|       |-- frame_0353.jpg
|       |-- frame_0354.jpg
|       |-- frame_0355.jpg
|       |-- frame_0356.jpg
|       |-- frame_0357.jpg
|       |-- frame_0358.jpg
|       |-- frame_0359.jpg
|       |-- frame_0360.jpg
|       |-- frame_0361.jpg
|       |-- frame_0362.jpg
|       |-- frame_0363.jpg
|       |-- frame_0364.jpg
|       |-- frame_0365.jpg
|       |-- frame_0366.jpg
|       |-- frame_0367.jpg
|       |-- frame_0368.jpg
|       |-- frame_0369.jpg
|       |-- frame_0370.jpg
|       |-- frame_0371.jpg
|       |-- frame_0372.jpg
|       |-- frame_0373.jpg
|       |-- frame_0374.jpg
|       |-- frame_0375.jpg
|       |-- frame_0376.jpg
|       |-- frame_0377.jpg
|       |-- frame_0378.jpg
|       |-- frame_0379.jpg
|       |-- frame_0380.jpg
|       |-- frame_0381.jpg
|       |-- frame_0382.jpg
|       |-- frame_0383.jpg
|       |-- frame_0384.jpg
|       |-- frame_0385.jpg
|       |-- frame_0386.jpg
|       |-- frame_0387.jpg
|       |-- frame_0388.jpg
|       |-- frame_0389.jpg
|       |-- frame_0390.jpg
|       |-- frame_0391.jpg
|       |-- frame_0392.jpg
|       |-- frame_0393.jpg
|       |-- frame_0394.jpg
|       |-- frame_0395.jpg
|       |-- frame_0396.jpg
|       |-- frame_0397.jpg
|       |-- frame_0398.jpg
|       |-- frame_0399.jpg
|       |-- frame_0400.jpg
|       |-- frame_0401.jpg
|       |-- frame_0402.jpg
|       |-- frame_0403.jpg
|       |-- frame_0404.jpg
|       |-- frame_0405.jpg
|       |-- frame_0406.jpg
|       |-- frame_0407.jpg
|       |-- frame_0408.jpg
|       |-- frame_0409.jpg
|       |-- frame_0410.jpg
|       |-- frame_0411.jpg
|       |-- frame_0412.jpg
|       |-- frame_0413.jpg
|       |-- frame_0414.jpg
|       |-- frame_0415.jpg
|       |-- frame_0416.jpg
|       |-- frame_0417.jpg
|       |-- frame_0418.jpg
|       |-- frame_0419.jpg
|       |-- frame_0420.jpg
|       |-- frame_0421.jpg
|       |-- frame_0422.jpg
|       |-- frame_0423.jpg
|       |-- frame_0424.jpg
|       |-- frame_0425.jpg
|       |-- frame_0426.jpg
|       |-- frame_0427.jpg
|       |-- frame_0428.jpg
|       |-- frame_0429.jpg
|       |-- frame_0430.jpg
|       |-- frame_0431.jpg
|       |-- frame_0432.jpg
|       |-- frame_0433.jpg
|       |-- frame_0434.jpg
|       |-- frame_0435.jpg
|       |-- frame_0436.jpg
|       |-- frame_0437.jpg
|       |-- frame_0438.jpg
|       |-- frame_0439.jpg
|       |-- frame_0440.jpg
|       |-- frame_0441.jpg
|       |-- frame_0442.jpg
|       |-- frame_0443.jpg
|       |-- frame_0444.jpg
|       |-- frame_0445.jpg
|       |-- frame_0446.jpg
|       |-- frame_0447.jpg
|       |-- frame_0448.jpg
|       |-- frame_0449.jpg
|       |-- frame_0450.jpg
|       |-- frame_0451.jpg
|       |-- frame_0452.jpg
|       |-- frame_0453.jpg
|       |-- frame_0454.jpg
|       |-- frame_0455.jpg
|       |-- frame_0456.jpg
|       |-- frame_0457.jpg
|       |-- frame_0458.jpg
|       |-- frame_0459.jpg
|       |-- frame_0460.jpg
|       |-- frame_0461.jpg
|       |-- frame_0462.jpg
|       |-- frame_0463.jpg
|       |-- frame_0464.jpg
|       |-- frame_0465.jpg
|       |-- frame_0466.jpg
|       |-- frame_0467.jpg
|       |-- frame_0468.jpg
|       |-- frame_0469.jpg
|       |-- frame_0470.jpg
|       |-- frame_0471.jpg
|       |-- frame_0472.jpg
|       `-- frame_0473.jpg
|-- evals
|   |-- eval7_scenes_dense.py
|   |-- eval_7scenes.sh
|   |-- eval_euroc.sh
|   |-- eval_tum.sh
|   |-- geometry_eval_utils.py
|   |-- __init__.py
|   |-- process_logs_7scenes.py
|   `-- process_logs_tum.py
|-- generate_project_doc.py
|-- LICENSE
|-- LOG
|   |-- 00_baseline.log
|   |-- 01_gate_only.log
|   |-- 02_loop_only.log
|   |-- 03_all_edges.log
|   `-- 04_full.log
|-- main.py
|-- office_loop.zip
|-- poses_logs
|   |-- 300.0.npz
|   |-- 302.0.npz
|   |-- 305.0.npz
|   |-- 307.0.npz
|   |-- 310.0.npz
|   |-- 314.0.npz
|   |-- 318.0.npz
|   |-- 322.0.npz
|   |-- 326.0.npz
|   |-- 329.0.npz
|   |-- 332.0.npz
|   |-- 334.0.npz
|   |-- 336.0.npz
|   |-- 338.0.npz
|   |-- 340.0.npz
|   |-- 342.0.npz
|   |-- 344.0.npz
|   |-- 346.0.npz
|   |-- 348.0.npz
|   |-- 351.0.npz
|   |-- 354.0.npz
|   |-- 356.0.npz
|   |-- 358.0.npz
|   |-- 360.0.npz
|   |-- 363.0.npz
|   |-- 366.0.npz
|   |-- 368.0.npz
|   |-- 370.0.npz
|   |-- 372.0.npz
|   |-- 374.0.npz
|   |-- 377.0.npz
|   |-- 379.0.npz
|   |-- 382.0.npz
|   |-- 384.0.npz
|   |-- 386.0.npz
|   |-- 388.0.npz
|   |-- 390.0.npz
|   |-- 391.0.npz
|   |-- 392.0.npz
|   |-- 394.0.npz
|   |-- 396.0.npz
|   |-- 397.0.npz
|   `-- 399.0.npz
|-- poses.txt
|-- README.md
|-- requirements.txt
|-- run_ablation_4groups.py
|-- run_chunked_vggt_slam.py
|-- RUNS
|   |-- ablation_office_loop_patch1
|   |   |-- all_runs.csv
|   |   |-- baseline
|   |   |-- edge_stats_all.csv
|   |   |-- gate_only
|   |   |-- gate_weight
|   |   `-- weight_only
|   `-- ablation_test
|       |-- all_runs.csv
|       |-- baseline
|       |-- edge_stats_all.csv
|       |-- gate_only
|       |-- gate_weight
|       `-- weight_only
|-- salad
|   |-- assets
|   |   |-- dino_salad_title.png
|   |   `-- method.jpg
|   |-- dataloaders
|   |   |-- GSVCitiesDataloader.py
|   |   |-- GSVCitiesDataset.py
|   |   |-- MapillaryDataset.py
|   |   |-- PittsburgDataset.py
|   |   `-- val
|   |-- datasets
|   |   |-- msls_test
|   |   |-- msls_val
|   |   |-- Nordland
|   |   |-- Pittsburgh
|   |   `-- SPED
|   |-- environment.yml
|   |-- hubconf.py
|   |-- LICENSE
|   |-- README.md
|   |-- salad
|   |   |-- eval.py
|   |   |-- __init__.py
|   |   |-- main.py
|   |   |-- models_salad
|   |   |-- utils
|   |   `-- vpr_model.py
|   `-- setup.py
|-- scripts
|   |-- align_points.py
|   |-- eval_corridor_extra_metrics.py
|   |-- locate_loop_entry.sh
|   |-- pylog.sh
|   |-- ros_to_jpg.py
|   |-- run_corridor_compare_plus.py
|   |-- run_corridor_compare.py
|   |-- run_semantic_suite_plus.py
|   |-- run_semantic_suite.py
|   `-- undistort.py
|-- setup.py
|-- tools
|   |-- run_ablation_corridor_semantic.sh
|   `-- sem_pretrain
|       |-- extract_dino_feats.py
|       |-- train_metric.py
|       `-- vis_retrieval.py
|-- vggt
|   |-- CODE_OF_CONDUCT.md
|   |-- CONTRIBUTING.md
|   |-- demo_colmap.py
|   |-- demo_gradio.py
|   |-- demo_viser.py
|   |-- docs
|   |   `-- package.md
|   |-- examples
|   |   |-- kitchen
|   |   |-- llff_fern
|   |   |-- llff_flower
|   |   |-- room
|   |   |-- single_cartoon
|   |   |-- single_oil_painting
|   |   `-- videos
|   |-- LICENSE.txt
|   |-- pyproject.toml
|   |-- README.md
|   |-- requirements_demo.txt
|   |-- requirements.txt
|   |-- training
|   |   |-- config
|   |   |-- data
|   |   |-- __init__.py
|   |   |-- launch.py
|   |   |-- loss.py
|   |   |-- README.md
|   |   |-- trainer.py
|   |   `-- train_utils
|   |-- vggt
|   |   |-- dependency
|   |   |-- heads
|   |   |-- layers
|   |   |-- models
|   |   `-- utils
|   `-- visual_util.py
`-- vggt_slam
    |-- frame_overlap.py
    |-- gradio_viewer.py
    |-- graph.py
    |-- graph_se3.py
    |-- h_solve.py
    |-- __init__.py
    |-- loop_closure.py
    |-- map.py
    |-- semantic_backend.py
    |-- slam_utils.py
    |-- solver.py
    `-- submap.py

55 directories, 703 files

```

## Python 代码文件

### app.py

```python
import os
import zipfile
import glob
import shutil
import numpy as np
import torch
import gradio as gr
import cv2
from tqdm import tqdm

import vggt_slam.slam_utils as utils
from vggt_slam.solver import Solver
from vggt.models.vggt import VGGT


def run_slam(
    image_zip,
    use_sim3=False,
    submap_size=16,
    max_loops=1,
    min_disparity=50.0,
    conf_threshold=25.0
):
    # Handle zip from Gradio
    zip_path = image_zip.name

    # Clean and extract
    tmp_dir = "temp_images"
    if os.path.exists(tmp_dir):
        shutil.rmtree(tmp_dir)
    os.makedirs(tmp_dir, exist_ok=True)

    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(tmp_dir)
        print("Extracted files:", zip_ref.namelist())

    # Recursive glob
    image_paths = [
        f for f in glob.glob(os.path.join(tmp_dir, "**", "*"), recursive=True)
        if "depth" not in f.lower() and f.lower().endswith((".jpg", ".png", ".jpeg"))
    ]

    image_paths = utils.sort_images_by_number(image_paths)

    use_optical_flow_downsample = True
    device = "cuda" if torch.cuda.is_available() else "cpu"

    solver = Solver(
        init_conf_threshold=conf_threshold,
        use_point_map=False,
        use_sim3=use_sim3,
        gradio_mode=True
    )

    model = VGGT()
    _URL = "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt"
    model.load_state_dict(torch.hub.load_state_dict_from_url(_URL))
    model.eval()
    model = model.to(device)

    image_names_subset = []
    for image_name in tqdm(image_paths):
        if use_optical_flow_downsample:
            img = cv2.imread(image_name)
            enough_disparity = solver.flow_tracker.compute_disparity(img, min_disparity, False)
            if enough_disparity:
                image_names_subset.append(image_name)
        else:
            image_names_subset.append(image_name)

        # Run submap
        if len(image_names_subset) == submap_size + 1 or image_name == image_paths[-1]:
            print(image_names_subset)
            predictions = solver.run_predictions(image_names_subset, model, max_loops)
            solver.add_points(predictions)
            solver.graph.optimize()
            solver.map.update_submap_homographies(solver.graph)

            image_names_subset = image_names_subset[-1:]

    solver.update_all_submap_vis()

    num_submaps = solver.map.get_num_submaps()
    num_loops = solver.graph.get_num_loops()
    glb_path = solver.export_3d_scene()
    message = f"VGGT-SLAM completed with {num_submaps} submaps and {num_loops} loop closures."
    return glb_path, message


# Setup Gradio outputs
model_output = gr.Model3D(label="🗺️ Reconstructed 3D Map", height=600)
status_output = gr.Textbox(label="", show_label=False) 

demo = gr.Interface(
    fn=run_slam,
    inputs=[
        gr.File(label="Upload .zip of images", file_types=[".zip"]),
        gr.Checkbox(label="Use Sim3", value=False),
        gr.Slider(4, 32, value=16, step=1, label="Submap Size"),
        gr.Slider(0, 5, value=1, step=1, label="Max potential loop closures to add for each new submap"),
        gr.Slider(0.0, 100.0, value=50.0, step=1.0, label="Minimum disparity between keyframes"),
        gr.Slider(0.0, 100.0, value=25.0, step=1.0, label="Confidence Threshold (increasing will decrease number of points)"),
    ],
    outputs=[model_output, status_output],
    examples=[["office_loop.zip"]],
    title="VGGT-SLAM Demo",
    description=(
        "We've prepared a simple demo of VGGT-SLAM on Hugging Face with an example scene. Our github repo contains a more powerful visualization which shows incremental scene reconstruction. \n\n"
        "To try your own scene, upload a ZIP of RGB images and run VGGT-SLAM to reconstruct the scene.\n\n"
        "Outputs a 3D point cloud and estimated camera poses.\n\n"
        "⏱️ **Estimated runtime for provided example scene: <1 minute**"
    ),
    allow_flagging="never",
    theme="default"
)

if __name__ == "__main__":
    demo.launch()

```

### generate_project_doc.py

```python
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Export project tree + source files into a single Word (.docx).

Usage:
  python export_project_to_word.py --root . --out project_dump.docx
  python export_project_to_word.py --root . --out dump.docx --ext .py .yaml .yml .json .sh
"""

import os
import sys
import argparse
from datetime import datetime
from pathlib import Path

from docx import Document
from docx.shared import Pt
from docx.oxml.ns import qn


DEFAULT_EXCLUDE_DIRS = {
    ".git", ".hg", ".svn",
    "__pycache__", ".mypy_cache", ".pytest_cache",
    ".idea", ".vscode",
    "venv", ".venv", "env", ".env",
    "build", "dist", ".eggs",
    "logs", "log", "outputs", "output", "results", "runs",
    "data", "datasets", "assets", "tmp", "cache",
    "checkpoints", "weights",
}

DEFAULT_EXCLUDE_FILE_EXTS = {
    ".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp",
    ".mp4", ".mov", ".avi", ".mkv",
    ".pt", ".pth", ".ckpt", ".onnx", ".tflite",
    ".npz", ".npy",
    ".zip", ".tar", ".gz", ".7z", ".rar",
    ".so", ".dll", ".dylib",
    ".pdf",
}

DEFAULT_INCLUDE_FILES = {
    "pyproject.toml", "requirements.txt", "environment.yml",
    "setup.py", "setup.cfg", "Pipfile", "Pipfile.lock",
    "README.md", "README.rst",
}


def set_monospace(run, font_name="Consolas", font_size=9):
    run.font.name = font_name
    run._element.rPr.rFonts.set(qn("w:eastAsia"), font_name)  # for CJK
    run.font.size = Pt(font_size)


def add_code_block(doc: Document, text: str):
    """
    Insert a preformatted-like block: single paragraph, monospace, line breaks preserved.
    """
    p = doc.add_paragraph()
    r = p.add_run()
    set_monospace(r, font_name="Consolas", font_size=9)
    # Preserve newlines
    lines = text.splitlines()
    for i, line in enumerate(lines):
        if i > 0:
            r.add_break()  # line break
        r.add_text(line)


def build_tree(root: Path, exclude_dirs: set, max_entries: int = 20000) -> str:
    """
    Build a 'tree'-like text without calling external commands.
    """
    root = root.resolve()
    lines = [str(root)]
    entries_count = 0

    def is_excluded_dir(p: Path) -> bool:
        return p.name in exclude_dirs

    def walk_dir(current: Path, prefix: str = ""):
        nonlocal entries_count
        if entries_count >= max_entries:
            lines.append(prefix + "└── [TRUNCATED: too many entries]")
            return

        try:
            children = sorted(list(current.iterdir()), key=lambda x: (x.is_file(), x.name.lower()))
        except PermissionError:
            lines.append(prefix + "└── [PermissionError]")
            return

        # filter excluded dirs
        filtered = []
        for c in children:
            if c.is_dir() and is_excluded_dir(c):
                continue
            filtered.append(c)

        for idx, c in enumerate(filtered):
            if entries_count >= max_entries:
                lines.append(prefix + "└── [TRUNCATED: too many entries]")
                return
            entries_count += 1

            is_last = (idx == len(filtered) - 1)
            connector = "└── " if is_last else "├── "
            lines.append(prefix + connector + c.name)

            if c.is_dir():
                extension = "    " if is_last else "│   "
                walk_dir(c, prefix + extension)

    walk_dir(root, "")
    return "\n".join(lines)


def should_include_file(path: Path, exts: set, exclude_file_exts: set) -> bool:
    if path.name in DEFAULT_INCLUDE_FILES:
        return True
    if path.suffix.lower() in exclude_file_exts:
        return False
    return path.suffix.lower() in exts


def iter_files(root: Path, exclude_dirs: set, exts: set, exclude_file_exts: set):
    root = root.resolve()
    for dirpath, dirnames, filenames in os.walk(root):
        # prune dirs in-place
        dirnames[:] = [d for d in dirnames if d not in exclude_dirs]
        for fn in sorted(filenames):
            p = Path(dirpath) / fn
            if should_include_file(p, exts, exclude_file_exts):
                yield p


def read_text_file(path: Path, max_bytes: int, max_lines: int):
    data = path.read_bytes()
    truncated = False

    if len(data) > max_bytes:
        data = data[:max_bytes]
        truncated = True

    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        text = data.decode("utf-8", errors="replace")

    lines = text.splitlines()
    if len(lines) > max_lines:
        lines = lines[:max_lines]
        truncated = True

    return "\n".join(lines), truncated


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, default=".", help="project root")
    ap.add_argument("--out", type=str, default="project_dump.docx", help="output docx path")
    ap.add_argument("--ext", nargs="*", default=[".py"], help="extensions to include, e.g. .py .yaml .yml")
    ap.add_argument("--exclude-dirs", nargs="*", default=sorted(DEFAULT_EXCLUDE_DIRS), help="dir names to exclude")
    ap.add_argument("--max-bytes", type=int, default=300_000, help="max bytes per file before truncation")
    ap.add_argument("--max-lines", type=int, default=5000, help="max lines per file before truncation")
    ap.add_argument("--max-tree-entries", type=int, default=20000, help="cap project tree entries")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    out_path = Path(args.out).resolve()

    exts = {e if e.startswith(".") else ("." + e) for e in args.ext}
    exclude_dirs = set(args.exclude_dirs)

    doc = Document()

    # Title
    doc.add_heading(f"Project Dump: {root.name}", level=0)
    doc.add_paragraph(f"Root: {root}")
    doc.add_paragraph(f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    doc.add_paragraph(f"Included extensions: {', '.join(sorted(exts))}")
    doc.add_paragraph(f"Excluded dirs: {', '.join(sorted(exclude_dirs))}")
    doc.add_paragraph(f"Per-file truncation: max_bytes={args.max_bytes}, max_lines={args.max_lines}")

    # Tree
    doc.add_heading("Project Tree", level=1)
    tree_text = build_tree(root, exclude_dirs=exclude_dirs, max_entries=args.max_tree_entries)
    add_code_block(doc, tree_text)

    # Files
    doc.add_heading("Files", level=1)
    file_count = 0
    total_bytes = 0

    for p in iter_files(root, exclude_dirs=exclude_dirs, exts=exts, exclude_file_exts=DEFAULT_EXCLUDE_FILE_EXTS):
        rel = p.relative_to(root)
        try:
            size = p.stat().st_size
        except OSError:
            continue

        file_count += 1
        total_bytes += size

        doc.add_heading(str(rel), level=2)
        doc.add_paragraph(f"Size: {size} bytes")

        try:
            content, truncated = read_text_file(p, max_bytes=args.max_bytes, max_lines=args.max_lines)
        except Exception as e:
            doc.add_paragraph(f"[READ ERROR] {type(e).__name__}: {e}")
            continue

        if truncated:
            doc.add_paragraph("[TRUNCATED] content was cut by size/line limits.")

        add_code_block(doc, content)

    # Summary
    doc.add_heading("Summary", level=1)
    doc.add_paragraph(f"Included files: {file_count}")
    doc.add_paragraph(f"Total raw size (before truncation): {total_bytes} bytes")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    doc.save(str(out_path))
    print(f"[OK] Wrote: {out_path}")


if __name__ == "__main__":
    main()

```

### main.py

```python
import os
import glob
import argparse

import numpy as np
import torch
from tqdm.auto import tqdm
import cv2
import matplotlib.pyplot as plt

import vggt_slam.slam_utils as utils
from vggt_slam.solver import Solver
from vggt.models.vggt import VGGT

parser = argparse.ArgumentParser(description="VGGT-SLAM demo")
parser.add_argument("--image_folder", type=str, default="examples/kitchen/images/", help="Path to folder containing images")
parser.add_argument("--vis_map", action="store_true", help="Visualize point cloud in viser as it is being build, otherwise only show the final map")
parser.add_argument("--vis_flow", action="store_true", help="Visualize optical flow from RAFT for keyframe selection")
parser.add_argument("--log_results", action="store_true", help="save txt file with results")
parser.add_argument("--skip_dense_log", action="store_true", help="by default, logging poses and logs dense point clouds. If this flag is set, dense logging is skipped")
parser.add_argument("--log_path", type=str, default="poses.txt", help="Path to save the log file")

parser.add_argument("--use_sim3", action="store_true", help="Use Sim3 instead of SL(4)")
parser.add_argument("--plot_focal_lengths", action="store_true", help="Plot focal lengths for the submaps")
parser.add_argument("--submap_size", type=int, default=16, help="Number of new frames per submap, does not include overlapping frames or loop closure frames")
parser.add_argument("--overlapping_window_size", type=int, default=1, help="ONLY DEFAULT OF 1 SUPPORTED RIGHT NOW. Number of overlapping frames, which are used in SL(4) estimation")
parser.add_argument("--downsample_factor", type=int, default=1, help="Factor to reduce image size by 1/N")
parser.add_argument("--max_loops", type=int, default=1, help="Maximum number of loop closures per submap")
parser.add_argument("--min_disparity", type=float, default=50, help="Minimum disparity to generate a new keyframe")
parser.add_argument("--use_point_map", action="store_true", help="Use point map instead of depth-based points")
parser.add_argument("--conf_threshold", type=float, default=25.0, help="Initial percentage of low-confidence points to filter out")
parser.add_argument("--vis_stride", type=int, default=1, help="Stride interval in the 3D point cloud image for visualization.")
parser.add_argument("--vis_point_size", type=float, default=0.003, help="Visualization point size")

# --- semantic backend ---
parser.add_argument("--use_semantic_backend", action="store_true", help="Enable semantic backend.")
parser.add_argument("--semantic_backend_cfg", type=str, default="", help="Optional semantic backend config path.")
parser.add_argument("--semantic_min_sim", type=float, default=0.25, help="Semantic similarity threshold.")

# --- semantic gate (ablation) ---
# 兼容旧参数：filter_loops / gate_retrieved
# Solver 支持：off / filter / retrieved / both
parser.add_argument(
    "--semantic_gate_mode",
    type=str,
    default="both",
    choices=["off", "filter", "retrieved", "both", "filter_loops", "gate_retrieved"],
    help="Semantic gate mode: off|filter|retrieved|both (legacy: filter_loops|gate_retrieved)",
)

# ✅ 兼容你现在脚本在传的开关：--disable_semantic_gate
parser.add_argument("--disable_semantic_gate", action="store_true",
                    help="Alias: disable semantic gate/filter even if semantic backend enabled.")

# ✅ 兼容你现在脚本在传的参数：--edge_stats_path
parser.add_argument("--edge_stats_path", type=str, default="",
                    help="Optional: path to save edge stats csv (sim/w/n_good...). Empty = disable.")

# --- semantic factor reweighting ---
parser.add_argument("--semantic_weight_mode", type=str, default="loop_only",
                    choices=["off", "loop_only", "all_edges"],
                    help="semantic factor reweighting: off|loop_only|all_edges")
parser.add_argument("--semantic_w_min", type=float, default=0.25)
parser.add_argument("--semantic_w_max", type=float, default=4.0)
parser.add_argument("--semantic_w_s0", type=float, default=0.25, help="similarity threshold start")
parser.add_argument("--semantic_w_gamma", type=float, default=2.0, help="weight mapping curve")
parser.add_argument("--semantic_w_degen_beta", type=float, default=0.0, help="degeneracy boost beta (0=off)")
parser.add_argument("--semantic_w_degen_ref_good", type=int, default=2000, help="ref good_mask size for degeneracy")

# --- corridor uniqueness down-weighting (NEW) ---
parser.add_argument("--semantic_u_enable", action="store_true",
                    help="Enable corridor uniqueness down-weighting (recommended for long similar corridors).")
parser.add_argument("--semantic_u_topk_submaps", type=int, default=8,
                    help="How many recent submaps to use for uniqueness estimation.")
parser.add_argument("--semantic_u_m0", type=float, default=0.05,
                    help="Scale for margin->u. smaller=more sensitive down-weighting.")
parser.add_argument("--semantic_u_min", type=float, default=0.25,
                    help="Lower bound of uniqueness factor u in [u_min, 1].")

parser.add_argument("--disable_loop_closure", action="store_true", help="Disable loop closure entirely (ablation)")


def _normalize_semantic_gate_mode(mode: str, disable: bool) -> str:
    """把 legacy gate_mode 映射到 Solver 支持的 off/filter/retrieved/both。"""
    if disable:
        return "off"
    m = (mode or "").strip().lower()
    if m in ("filter_loops", "filterloop", "filterloops"):
        return "filter"
    if m in ("gate_retrieved", "gateretrieved"):
        return "retrieved"
    if m in ("off", "filter", "retrieved", "both"):
        return m
    return "both"


def main():
    args = parser.parse_args()

    # 兼容：disable_semantic_gate 覆盖 semantic_gate_mode；并把 legacy 名字映射到 Solver 接受的名字
    args.semantic_gate_mode = _normalize_semantic_gate_mode(args.semantic_gate_mode, args.disable_semantic_gate)

    use_optical_flow_downsample = True
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    solver = Solver(
        init_conf_threshold=args.conf_threshold,
        use_point_map=args.use_point_map,
        use_sim3=args.use_sim3,
        gradio_mode=False,
        vis_stride=args.vis_stride,
        vis_point_size=args.vis_point_size,

        use_semantic_backend=args.use_semantic_backend,
        semantic_backend_cfg=args.semantic_backend_cfg,
        semantic_min_sim=args.semantic_min_sim,

        semantic_gate_mode=args.semantic_gate_mode,
        disable_semantic_gate=args.disable_semantic_gate,

        edge_stats_path=args.edge_stats_path,

        semantic_weight_mode=args.semantic_weight_mode,
        semantic_w_min=args.semantic_w_min,
        semantic_w_max=args.semantic_w_max,
        semantic_w_s0=args.semantic_w_s0,
        semantic_w_gamma=args.semantic_w_gamma,
        semantic_w_degen_beta=args.semantic_w_degen_beta,
        semantic_w_degen_ref_good=args.semantic_w_degen_ref_good,

        # NEW: corridor uniqueness
        semantic_u_enable=args.semantic_u_enable,
        semantic_u_topk_submaps=args.semantic_u_topk_submaps,
        semantic_u_m0=args.semantic_u_m0,
        semantic_u_min=args.semantic_u_min,
    )

    print("Initializing and loading VGGT model...")
    model = VGGT()
    _URL = "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt"
    model.load_state_dict(torch.hub.load_state_dict_from_url(_URL))
    model.eval()
    model = model.to(device)

    print(f"Loading images from {args.image_folder}...")
    image_names = [
        f for f in glob.glob(os.path.join(args.image_folder, "*"))
        if "depth" not in os.path.basename(f).lower()
        and "txt" not in os.path.basename(f).lower()
        and "db" not in os.path.basename(f).lower()
    ]
    image_names = utils.sort_images_by_number(image_names)
    image_names = utils.downsample_images(image_names, args.downsample_factor)
    print(f"Found {len(image_names)} images")

    image_names_subset = []
    data = []

    for image_name in tqdm(image_names):
        if use_optical_flow_downsample:
            img = cv2.imread(image_name)
            enough_disparity = solver.flow_tracker.compute_disparity(img, args.min_disparity, args.vis_flow)
            if enough_disparity:
                image_names_subset.append(image_name)
        else:
            image_names_subset.append(image_name)

        if len(image_names_subset) == args.submap_size + args.overlapping_window_size or image_name == image_names[-1]:
            print(image_names_subset)
            max_loops_eff = 0 if args.disable_loop_closure else args.max_loops

            predictions = solver.run_predictions(image_names_subset, model, max_loops_eff)
            data.append(predictions["intrinsic"][:, 0, 0])

            solver.add_points(predictions)
            solver.graph.debug_optimize_incremental()
            solver.map.update_submap_homographies(solver.graph)

            loop_closure_detected = len(predictions.get("detected_loops", [])) > 0
            if args.vis_map:
                if loop_closure_detected:
                    solver.update_all_submap_vis()
                else:
                    solver.update_latest_submap_vis()

            image_names_subset = image_names_subset[-args.overlapping_window_size:]

    print("Total number of submaps in map", solver.map.get_num_submaps())
    print("Total number of loop closures in map", solver.graph.get_num_loops())

    if not args.vis_map:
        solver.update_all_submap_vis()

    if args.log_results:
        if os.path.isdir(args.log_path):
            args.log_path = os.path.join(args.log_path, "poses.txt")
        parent = os.path.dirname(args.log_path)
        if parent:
            os.makedirs(parent, exist_ok=True)

        solver.map.write_poses_to_file(args.log_path)

        if not args.skip_dense_log:
            solver.map.save_framewise_pointclouds(args.log_path.replace(".txt", "_logs"))

    if args.plot_focal_lengths:
        colors = plt.cm.viridis(np.linspace(0, 1, len(data)))
        plt.figure(figsize=(8, 6))
        for i, values in enumerate(data):
            y = values
            x = [i] * len(values)
            plt.scatter(x, y, color=colors[i], label=f"List {i+1}")
        plt.xlabel("poses")
        plt.ylabel("Focal lengths")
        plt.grid()
        plt.show()


if __name__ == "__main__":
    main()

```

### run_ablation_4groups.py

```python
# run_ablation_4groups.py
# 用法示例：
# python run_ablation_4groups.py \
#   --data /media/omnisky/18/cst1/project/VGGT-SLAM/DATA/corridor_0300-0399 \
#   --out_dir /media/omnisky/18/cst1/project/VGGT-SLAM/RUNS/ablation_20260116 \
#   --submap_size 16 --max_loops 1 \
#   --min_disparity 50 --conf_threshold 25 \
#   --semantic_backend_cfg /path/to/semantic_backend.yaml \
#   --semantic_min_sim 0.25
#
# 产物：
# out_dir/
#   all_runs.csv
#   edge_stats_all.csv
#   baseline/ edge_stats.csv poses.npy run_meta.json
#   gate_only/ ...
#   weight_only/ ...
#   gate_weight/ ...

from __future__ import annotations

import os
import re
import csv
import time
import json
import glob
import shutil
import zipfile
import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import cv2

from vggt_slam.solver import Solver
from vggt.models.vggt import VGGT


# -----------------------------
# utils: list & sort images
# -----------------------------
def _extract_numeric_key(p: str) -> Tuple:
    name = os.path.basename(p)
    m = re.search(r"(\d+(?:\.\d+)?)", name)
    if m:
        return (0, float(m.group(1)), name)
    return (1, name)


def list_images(data_path: str, tmp_dir: str) -> List[str]:
    """
    data_path 支持：
      - 文件夹：递归找 jpg/png/jpeg
      - zip：解压到 tmp_dir 后递归找图
    """
    p = Path(data_path)
    if p.is_file() and p.suffix.lower() == ".zip":
        if os.path.exists(tmp_dir):
            shutil.rmtree(tmp_dir)
        os.makedirs(tmp_dir, exist_ok=True)
        with zipfile.ZipFile(str(p), "r") as zf:
            zf.extractall(tmp_dir)
        root = tmp_dir
    else:
        root = str(p)

    imgs = [
        f for f in glob.glob(os.path.join(root, "**", "*"), recursive=True)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
        and "depth" not in os.path.basename(f).lower()
    ]
    imgs.sort(key=_extract_numeric_key)
    return imgs


# -----------------------------
# trajectory & metrics
# -----------------------------
def collect_trajectory_from_map(solver: Solver) -> Tuple[np.ndarray, List[float]]:
    """
    收集所有 submap 的非 loop 帧世界位姿（4x4），按 frame_id 排序并去重。
    """
    submaps = solver.map.get_submaps()

    id2pose: Dict[float, np.ndarray] = {}
    id_list: List[float] = []

    for sm in submaps:
        poses_world = sm.get_all_poses_world(ignore_loop_closure_frames=True)  # (K,4,4)
        fids = sm.get_frame_ids()  # 不含 loop 帧

        if fids is None:
            continue

        K = min(len(fids), poses_world.shape[0])
        for i in range(K):
            fid = float(fids[i])
            if fid not in id2pose:
                id2pose[fid] = poses_world[i].copy()
                id_list.append(fid)

    id_list.sort()
    poses = np.stack([id2pose[fid] for fid in id_list], axis=0) if len(id_list) > 0 else np.zeros((0, 4, 4))
    return poses, id_list


def compute_pose_metrics(poses: np.ndarray) -> Dict[str, float]:
    """
    无 GT 的稳定可比较指标：
      - path_length_m：相邻位姿平移增量之和
      - loop_trans_err_m：起点到终点位移（回环序列越小越好；直行序列接近 path_length 很正常）
      - drift_ratio：loop_trans_err_m / path_length_m（回环序列越小越好）
    """
    n = int(poses.shape[0])
    if n < 2:
        return dict(
            n_poses=float(n),
            path_length_m=0.0,
            loop_trans_err_m=0.0,
            drift_ratio=0.0,
            step_trans_mean_m=0.0,
            step_trans_p95_m=0.0,
            step_trans_max_m=0.0,
        )

    t = poses[:, 0:3, 3]
    steps = np.linalg.norm(t[1:] - t[:-1], axis=1)
    path_len = float(np.sum(steps))
    loop_err = float(np.linalg.norm(t[-1] - t[0]))
    drift = float(loop_err / (path_len + 1e-9))

    return dict(
        n_poses=float(n),
        path_length_m=path_len,
        loop_trans_err_m=loop_err,
        drift_ratio=drift,
        step_trans_mean_m=float(np.mean(steps)),
        step_trans_p95_m=float(np.percentile(steps, 95)),
        step_trans_max_m=float(np.max(steps)),
    )


# -----------------------------
# edge_stats summarization
# -----------------------------
def _safe_float_series(values: List) -> np.ndarray:
    out = []
    for v in values:
        try:
            out.append(float(v))
        except Exception:
            out.append(np.nan)
    return np.asarray(out, dtype=np.float64)


def summarize_edge_stats(edge_stats_csv: str) -> Dict[str, float]:
    """
    从 edge_stats.csv 提取“论文级”统计，直接写进 all_runs.csv，便于对比。
    字段来自你 solver.py 的 EdgeStat：
      time,edge_type,src,dst,sim,w,n_good,mask_thr,mask_fallback_or,weight_mode,margin,u
    """
    p = Path(edge_stats_csv)
    if (not p.exists()) or p.stat().st_size == 0:
        return {}

    try:
        with p.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
    except Exception:
        return {}

    if len(rows) == 0:
        return {}

    out: Dict[str, float] = {}

    # 全部边数量
    out["n_edges_all"] = float(len(rows))

    def collect(edge_type: str) -> List[Dict]:
        return [r for r in rows if str(r.get("edge_type", "")).strip() == edge_type]

    for et in ["loop", "odom"]:
        d = collect(et)
        out[f"n_edges_{et}"] = float(len(d))

        if len(d) == 0:
            continue

        sim = _safe_float_series([r.get("sim", np.nan) for r in d])
        w = _safe_float_series([r.get("w", np.nan) for r in d])

        out[f"{et}_sim_mean"] = float(np.nanmean(sim))
        out[f"{et}_sim_p50"] = float(np.nanmedian(sim))
        out[f"{et}_w_mean"] = float(np.nanmean(w))
        out[f"{et}_w_min"] = float(np.nanmin(w))
        out[f"{et}_w_max"] = float(np.nanmax(w))

        # corridor anti-aliasing stats（如果有）
        if "margin" in d[0]:
            margin = _safe_float_series([r.get("margin", np.nan) for r in d])
            out[f"{et}_margin_mean"] = float(np.nanmean(margin))
            out[f"{et}_margin_p50"] = float(np.nanmedian(margin))
        if "u" in d[0]:
            u = _safe_float_series([r.get("u", np.nan) for r in d])
            out[f"{et}_u_mean"] = float(np.nanmean(u))
            out[f"{et}_u_p50"] = float(np.nanmedian(u))

        # 仅 odom：fallback_or 比例（几何退化信号）
        if et == "odom" and ("mask_fallback_or" in d[0]):
            fb = _safe_float_series([r.get("mask_fallback_or", 0) for r in d])
            out["odom_fallback_or_rate"] = float(np.mean(fb > 0))

        # 仅 odom：n_good（有效点数）统计（可选但很有用）
        if et == "odom" and ("n_good" in d[0]):
            ng = _safe_float_series([r.get("n_good", np.nan) for r in d])
            out["odom_n_good_mean"] = float(np.nanmean(ng))
            out["odom_n_good_p50"] = float(np.nanmedian(ng))

    return out


# -----------------------------
# core run
# -----------------------------
def build_solver(
    conf_threshold: float,
    use_point_map: bool,
    use_sim3: bool,
    use_semantic_backend: bool,
    semantic_backend_cfg: str,
    semantic_gate_mode: str,
    semantic_weight_mode: str,
    semantic_min_sim: float,
    edge_stats_path: str,
) -> Solver:
    solver = Solver(
        init_conf_threshold=float(conf_threshold),
        use_point_map=bool(use_point_map),
        use_sim3=bool(use_sim3),

        # 自动跑实验：避免 viser server 冲突/开端口
        gradio_mode=True,

        # semantic backend
        use_semantic_backend=bool(use_semantic_backend),
        semantic_backend_cfg=str(semantic_backend_cfg),
        semantic_min_sim=float(semantic_min_sim),

        # gate / weight
        semantic_gate_mode=str(semantic_gate_mode),
        semantic_weight_mode=str(semantic_weight_mode),

        # stats
        edge_stats_path=str(edge_stats_path),
    )
    return solver


def load_model(device: str, model_ckpt: str = "") -> torch.nn.Module:
    model = VGGT()
    if model_ckpt and os.path.exists(model_ckpt):
        sd = torch.load(model_ckpt, map_location="cpu")
        model.load_state_dict(sd)
    else:
        # 与 app.py 一致：从 URL 拉权重
        _URL = "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt"
        sd = torch.hub.load_state_dict_from_url(_URL, map_location="cpu")
        model.load_state_dict(sd)

    model.eval()
    model = model.to(device)
    return model


def warmup_model(model: torch.nn.Module, device: str):
    """
    可选：做一次短 warmup，减少 baseline first-run 慢导致的 seconds 偏差。
    """
    try:
        dummy = torch.zeros(1, 3, 224, 224, device=device, dtype=torch.float16)
        with torch.no_grad():
            _ = model(dummy)
    except Exception:
        pass


def run_one_group(
    group_name: str,
    image_paths: List[str],
    out_dir: str,
    model: torch.nn.Module,
    use_optical_flow_downsample: bool,
    min_disparity: float,
    submap_size: int,
    max_loops: int,
    conf_threshold: float,
    use_point_map: bool,
    use_sim3: bool,
    semantic_backend_cfg: str,
    semantic_min_sim: float,
    semantic_gate_mode: str,
    semantic_weight_mode: str,
    use_semantic_backend: bool,
    dataset_id: str,
    cmd_str: str,
) -> Dict[str, object]:

    gdir = Path(out_dir) / group_name
    gdir.mkdir(parents=True, exist_ok=True)

    edge_stats_path = str(gdir / "edge_stats.csv")
    poses_path = str(gdir / "poses.npy")
    meta_path = str(gdir / "run_meta.json")

    solver = build_solver(
        conf_threshold=conf_threshold,
        use_point_map=use_point_map,
        use_sim3=use_sim3,
        use_semantic_backend=use_semantic_backend,
        semantic_backend_cfg=semantic_backend_cfg,
        semantic_gate_mode=semantic_gate_mode,
        semantic_weight_mode=semantic_weight_mode,
        semantic_min_sim=semantic_min_sim,
        edge_stats_path=edge_stats_path,
    )

    subset: List[str] = []
    start = time.time()

    n_submaps = 0
    n_opt = 0
    n_selected = 0

    for idx, img_path in enumerate(image_paths):
        if use_optical_flow_downsample:
            img = cv2.imread(img_path)
            enough = solver.flow_tracker.compute_disparity(img, float(min_disparity), False)
            if enough:
                subset.append(img_path)
                n_selected += 1
        else:
            subset.append(img_path)
            n_selected += 1

        is_last = (idx == len(image_paths) - 1)
        if len(subset) == (int(submap_size) + 1) or is_last:
            if len(subset) < 2:
                subset = subset[-1:]
                continue

            predictions = solver.run_predictions(subset, model, int(max_loops))
            solver.add_points(predictions)
            solver.graph.optimize()
            solver.map.update_submap_homographies(solver.graph)

            n_submaps += 1
            n_opt += 1

            # overlap：保留最后一帧
            subset = subset[-1:]

    seconds = float(time.time() - start)

    poses, frame_ids = collect_trajectory_from_map(solver)
    np.save(poses_path, poses)

    metrics = compute_pose_metrics(poses)
    edge_summary = summarize_edge_stats(edge_stats_path)

    # 关闭 edge_stats 文件句柄（建议）
    try:
        fp = getattr(solver, "_edge_stats_fp", None)
        if fp is not None:
            fp.close()
    except Exception:
        pass

    meta = {
        "time": datetime.now().isoformat(timespec="seconds"),
        "group": group_name,
        "dataset": dataset_id,
        "n_images_total": int(len(image_paths)),
        "n_images_selected": int(n_selected),
        "n_submaps": int(n_submaps),
        "seconds": seconds,
        "poses_path": poses_path,
        "edge_stats_path": edge_stats_path,
        "cmd": cmd_str,
        "semantic": {
            "use_semantic_backend": bool(use_semantic_backend),
            "semantic_backend_cfg": str(semantic_backend_cfg),
            "semantic_min_sim": float(semantic_min_sim),
            "semantic_gate_mode": str(semantic_gate_mode),
            "semantic_weight_mode": str(semantic_weight_mode),
        },
        "metrics": metrics,
        "edge_summary": edge_summary,
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    row: Dict[str, object] = {
        "time": meta["time"],
        "group": group_name,
        "returncode": 0,
        "seconds": seconds,
        "dataset": dataset_id,
        "data": str(Path(out_dir).resolve()),
        "run_dir": str(gdir.resolve()),
        "poses_file": poses_path,
        "edge_stats_file": edge_stats_path,
        "n_images_total": int(len(image_paths)),
        "n_images_selected": int(n_selected),

        **metrics,

        "semantic_gate_mode": str(semantic_gate_mode),
        "semantic_weight_mode": str(semantic_weight_mode),
        "use_semantic_backend": int(bool(use_semantic_backend)),

        # 关键：把 loop/odom 的证据直接写进 all_runs
        **edge_summary,

        "cmd": cmd_str,
    }
    return row


def merge_edge_stats(out_dir: str, groups: List[str]) -> str:
    """
    合并各组 edge_stats.csv -> edge_stats_all.csv，并加 group 列
    """
    out_path = str(Path(out_dir) / "edge_stats_all.csv")
    rows = []
    header: Optional[List[str]] = None

    for g in groups:
        p = Path(out_dir) / g / "edge_stats.csv"
        if (not p.exists()) or p.stat().st_size == 0:
            continue
        with p.open("r", encoding="utf-8") as f:
            r = csv.DictReader(f)
            if header is None:
                header = ["group"] + list(r.fieldnames or [])
            for item in r:
                item2 = {"group": g}
                item2.update(item)
                rows.append(item2)

    if header is None:
        with open(out_path, "w", encoding="utf-8", newline="") as f:
            f.write("")
        return out_path

    with open(out_path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header)
        w.writeheader()
        for it in rows:
            w.writerow(it)

    return out_path


def write_all_runs_csv(out_dir: str, rows: List[Dict[str, object]]) -> str:
    """
    写 all_runs.csv：字段用“所有 row 的 key 并集”，保证不会丢字段。
    """
    out_path = str(Path(out_dir) / "all_runs.csv")
    keys = []
    seen = set()
    # 让重要字段排前面
    preferred = [
        "time","group","returncode","seconds","dataset","data","run_dir",
        "poses_file","edge_stats_file","n_images_total","n_images_selected",
        "n_poses","path_length_m","loop_trans_err_m","drift_ratio",
        "step_trans_mean_m","step_trans_p95_m","step_trans_max_m",
        "semantic_gate_mode","semantic_weight_mode","use_semantic_backend",
        "n_edges_all","n_edges_odom","n_edges_loop",
        "loop_sim_mean","loop_sim_p50","loop_w_mean","loop_w_min","loop_w_max",
        "odom_sim_mean","odom_sim_p50","odom_w_mean","odom_w_min","odom_w_max",
        "odom_fallback_or_rate","odom_n_good_mean","odom_n_good_p50",
        "odom_margin_mean","odom_margin_p50","odom_u_mean","odom_u_p50",
        "loop_margin_mean","loop_margin_p50","loop_u_mean","loop_u_p50",
        "cmd",
    ]
    for k in preferred:
        if any(k in r for r in rows):
            keys.append(k); seen.add(k)
    for r in rows:
        for k in r.keys():
            if k not in seen:
                keys.append(k); seen.add(k)

    with open(out_path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    return out_path


def build_cmd_str(args: argparse.Namespace) -> str:
    # 记录本次运行命令（方便复现实验）
    kv = vars(args).copy()
    # 删掉可能很长的东西也行，这里保留
    return "python run_ablation_4groups.py " + " ".join([f"--{k} {kv[k]}" if not isinstance(kv[k], bool) else (f"--{k}" if kv[k] else "") for k in kv]).strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, required=True, help="image folder OR .zip")
    ap.add_argument("--out_dir", type=str, required=True, help="output directory")
    ap.add_argument("--dataset_id", type=str, default="", help="写进CSV的dataset标识（不填则用 data 路径）")

    ap.add_argument("--submap_size", type=int, default=16)
    ap.add_argument("--max_loops", type=int, default=1)
    ap.add_argument("--min_disparity", type=float, default=50.0)
    ap.add_argument("--conf_threshold", type=float, default=25.0)
    ap.add_argument("--use_point_map", type=int, default=0)
    ap.add_argument("--use_sim3", type=int, default=0)
    ap.add_argument("--no_flow_downsample", action="store_true")

    ap.add_argument("--semantic_backend_cfg", type=str, default="", help="semantic backend cfg path")
    ap.add_argument("--semantic_min_sim", type=float, default=0.25)

    ap.add_argument("--model_ckpt", type=str, default="", help="local model.pt (optional)")
    ap.add_argument("--warmup", action="store_true", help="do a tiny model warmup to reduce first-run bias")
    args = ap.parse_args()

    out_dir = str(Path(args.out_dir).resolve())
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    dataset_id = args.dataset_id.strip() if args.dataset_id.strip() else str(Path(args.data).resolve())
    tmp_dir = str(Path(out_dir) / "_tmp_images")

    image_paths = list_images(args.data, tmp_dir=tmp_dir)
    if len(image_paths) == 0:
        raise RuntimeError(f"No images found from: {args.data}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(device=device, model_ckpt=args.model_ckpt)

    if args.warmup:
        warmup_model(model, device=device)

    # 四组定义（固定）
    groups = [
        ("baseline",     dict(use_semantic_backend=False, semantic_gate_mode="off",  semantic_weight_mode="off")),
        ("gate_only",    dict(use_semantic_backend=True,  semantic_gate_mode="both", semantic_weight_mode="off")),
        ("weight_only",  dict(use_semantic_backend=True,  semantic_gate_mode="off",  semantic_weight_mode="loop_only")),
        ("gate_weight",  dict(use_semantic_backend=True,  semantic_gate_mode="both", semantic_weight_mode="loop_only")),
    ]

    cmd_str = build_cmd_str(args)

    rows: List[Dict[str, object]] = []
    for gname, cfg in groups:
        print(f"\n==================== RUN {gname} ====================\n")
        try:
            row = run_one_group(
                group_name=gname,
                image_paths=image_paths,
                out_dir=out_dir,
                model=model,
                use_optical_flow_downsample=(not args.no_flow_downsample),
                min_disparity=float(args.min_disparity),
                submap_size=int(args.submap_size),
                max_loops=int(args.max_loops),
                conf_threshold=float(args.conf_threshold),
                use_point_map=bool(int(args.use_point_map)),
                use_sim3=bool(int(args.use_sim3)),
                semantic_backend_cfg=str(args.semantic_backend_cfg),
                semantic_min_sim=float(args.semantic_min_sim),
                semantic_gate_mode=str(cfg["semantic_gate_mode"]),
                semantic_weight_mode=str(cfg["semantic_weight_mode"]),
                use_semantic_backend=bool(cfg["use_semantic_backend"]),
                dataset_id=dataset_id,
                cmd_str=cmd_str,
            )
        except Exception as e:
            # 不中断其它组，写一行失败记录
            row = {
                "time": datetime.now().isoformat(timespec="seconds"),
                "group": gname,
                "returncode": 1,
                "seconds": 0.0,
                "dataset": dataset_id,
                "data": str(Path(out_dir).resolve()),
                "run_dir": str((Path(out_dir) / gname).resolve()),
                "poses_file": "",
                "edge_stats_file": "",
                "error": repr(e),
                "semantic_gate_mode": str(cfg["semantic_gate_mode"]),
                "semantic_weight_mode": str(cfg["semantic_weight_mode"]),
                "use_semantic_backend": int(bool(cfg["use_semantic_backend"])),
                "cmd": cmd_str,
            }
            print(f"[ERROR] group={gname} failed: {e}")

        rows.append(row)

    all_runs = write_all_runs_csv(out_dir, rows)
    edge_all = merge_edge_stats(out_dir, [g for g, _ in groups])

    print("\n==================== DONE ====================")
    print("all_runs.csv     :", all_runs)
    print("edge_stats_all.csv:", edge_all)


if __name__ == "__main__":
    main()

```

### run_chunked_vggt_slam.py

```python
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Chunked runner + visualization for VGGT-SLAM.

Features:
  1) Natural-sort images in --image_folder
  2) Split into chunks (default 100)
  3) Create per-chunk folder with images (symlink by default; fallback to copy)
  4) Run: python main.py --image_folder <chunk_images> + passthrough args
     - stream stdout in real-time to terminal AND write to run.log
  5) Compute simple texture/quality stats per chunk
  6) (Optional) Save per-chunk input visualization (RGB + gradient magnitude)
  7) Write chunk_summary.csv and summary plots (line plot + scatter)

IMPORTANT:
  - Wrapper args (like --viz) are NOT passed to main.py.
  - Only arguments after `--` are passed to main.py.
    Example:  ... --viz -- --max_loops 1

Usage:
  cd /path/to/VGGT-SLAM

  python run_chunked_vggt_slam.py \
    --repo_root . \
    --image_folder ./office_loop \
    --chunk_size 100 \
    --out_root runs_chunk100_viz \
    --prefer_symlink \
    --viz --samples_per_chunk 12 \
    -- --max_loops 1
"""

import argparse
import csv
import json
import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any

import numpy as np
from PIL import Image

# headless plotting (safe on servers)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def natural_key(s: str):
    """Natural sort key: frame_2 < frame_10"""
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]


def list_images(folder: Path) -> List[Path]:
    imgs = [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS]
    imgs.sort(key=lambda p: natural_key(p.name))
    return imgs


def safe_mkdir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def link_or_copy(src: Path, dst: Path, prefer_symlink: bool = True):
    """Symlink preferred; fallback to copy if symlink not permitted."""
    if dst.exists():
        return
    try:
        if prefer_symlink:
            os.symlink(src.resolve(), dst)
        else:
            shutil.copy2(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def load_gray_uint8(img_path: Path) -> np.ndarray:
    im = Image.open(img_path).convert("L")
    return np.asarray(im, dtype=np.uint8)


def texture_stats(img_paths: List[Path]) -> Dict[str, float]:
    """
    Simple texture/blur proxies (cheap, dependency-free):
      - lap_var_mean: variance of discrete Laplacian (higher => sharper/more texture)
      - grad_mean: mean abs gradient magnitude (higher => more edges)
      - brightness_mean/std
      - blur_score: inverse of lap_var_mean (higher => blurrier)
    """
    lap_vars, grad_means, means, stds = [], [], [], []

    for p in img_paths:
        g = load_gray_uint8(p).astype(np.float32)

        gx = np.abs(g[:, 1:] - g[:, :-1])
        gy = np.abs(g[1:, :] - g[:-1, :])
        grad_means.append(float(0.5 * (gx.mean() + gy.mean())))

        # 4-neighbor Laplacian: -4I + N+S+E+W
        lap = (-4.0 * g
               + np.roll(g, 1, axis=0) + np.roll(g, -1, axis=0)
               + np.roll(g, 1, axis=1) + np.roll(g, -1, axis=1))
        lap_vars.append(float(lap.var()))

        means.append(float(g.mean()))
        stds.append(float(g.std()))

    lap_var_mean = float(np.mean(lap_vars)) if lap_vars else 0.0
    return {
        "lap_var_mean": lap_var_mean,
        "grad_mean": float(np.mean(grad_means)) if grad_means else 0.0,
        "brightness_mean": float(np.mean(means)) if means else 0.0,
        "brightness_std": float(np.mean(stds)) if stds else 0.0,
        "blur_score": float(1.0 / (lap_var_mean + 1e-6)),
    }


def parse_generic_costs(log_text: str) -> Dict[str, Any]:
    """
    Best-effort: extract last numeric value from lines containing keywords.
    VGGT-SLAM may not print these; often returns found=False.
    """
    keywords = ["loss", "cost", "error", "chi2", "chi^2", "objective", "residual"]
    hits = []
    for ln in log_text.splitlines():
        low = ln.lower()
        if any(k in low for k in keywords):
            vals = re.findall(r"[-+]?\d*\.\d+|[-+]?\d+", ln)
            if vals:
                try:
                    hits.append(float(vals[-1]))
                except Exception:
                    pass
    if not hits:
        return {"found": False}
    return {
        "found": True,
        "count": len(hits),
        "last": float(hits[-1]),
        "min": float(np.min(hits)),
        "max": float(np.max(hits)),
        "mean": float(np.mean(hits)),
    }


def save_input_viz(img_paths: List[Path], out_path: Path, n_samples: int = 12, thumb_w: int = 320):
    """
    Save a contact sheet:
      Row 1: RGB
      Row 2: gradient magnitude (edge strength)
    """
    if not img_paths:
        return

    n = len(img_paths)
    n_samples = max(2, min(n_samples, n))
    idxs = np.linspace(0, n - 1, n_samples).round().astype(int).tolist()

    cols = min(6, n_samples)
    rows = int(np.ceil(n_samples / cols))

    fig = plt.figure(figsize=(cols * 3.2, rows * 3.4))

    for k, i in enumerate(idxs):
        p = img_paths[i]
        rgb = Image.open(p).convert("RGB")
        w, h = rgb.size
        scale = thumb_w / max(1, w)
        rgb = rgb.resize((int(w * scale), int(h * scale)))

        g = np.asarray(rgb.convert("L"), dtype=np.float32)
        gx = np.abs(g[:, 1:] - g[:, :-1])
        gy = np.abs(g[1:, :] - g[:-1, :])
        gx = np.pad(gx, ((0, 0), (0, 1)), mode="edge")
        gy = np.pad(gy, ((0, 1), (0, 0)), mode="edge")
        grad = (gx + gy) * 0.5
        grad = grad / (grad.max() + 1e-6)

        ax1 = plt.subplot(rows * 2, cols, k + 1)
        ax1.imshow(rgb)
        ax1.set_title(p.name, fontsize=8)
        ax1.axis("off")

        ax2 = plt.subplot(rows * 2, cols, cols * rows + k + 1)
        ax2.imshow(grad, cmap="gray")
        ax2.axis("off")

    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def save_summary_plots(summary_csv: Path, out_root: Path):
    """Generate summary plots for chunk texture proxies."""
    rows = []
    with summary_csv.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)
    if not rows:
        return

    chunk_id = np.array([int(r["chunk_id"]) for r in rows], dtype=int)
    lap = np.array([float(r["lap_var_mean"]) for r in rows], dtype=float)
    grad = np.array([float(r["grad_mean"]) for r in rows], dtype=float)
    blur = np.array([float(r["blur_score"]) for r in rows], dtype=float)

    # 1) Line plot over chunks
    fig1 = plt.figure(figsize=(10, 6))
    ax = plt.gca()
    ax.plot(chunk_id, lap, marker="o", label="lap_var_mean")
    ax.plot(chunk_id, grad, marker="o", label="grad_mean")
    ax.plot(chunk_id, blur, marker="o", label="blur_score")
    ax.set_xlabel("chunk_id")
    ax.set_ylabel("value")
    ax.set_title("Texture/Blur proxies over chunks")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig1.tight_layout()
    fig1.savefig(out_root / "summary_texture_over_chunks.png", dpi=160)
    plt.close(fig1)

    # 2) Scatter lap vs grad, annotate chunk ids
    fig2 = plt.figure(figsize=(7, 6))
    ax = plt.gca()
    ax.scatter(lap, grad)
    for i in range(len(chunk_id)):
        ax.annotate(str(chunk_id[i]), (lap[i], grad[i]), fontsize=9)
    ax.set_xlabel("lap_var_mean")
    ax.set_ylabel("grad_mean")
    ax.set_title("lap_var_mean vs grad_mean (annotated by chunk_id)")
    ax.grid(True, alpha=0.3)
    fig2.tight_layout()
    fig2.savefig(out_root / "scatter_lap_vs_grad.png", dpi=160)
    plt.close(fig2)


@dataclass
class ChunkResult:
    chunk_id: int
    start_idx: int
    end_idx: int
    n_images: int
    out_dir: Path
    ok: bool
    returncode: int
    seconds: float
    tex: Dict[str, float]
    costs: Dict[str, Any]


def strip_duplicate_image_folder_args(extra: List[str]) -> List[str]:
    """Remove accidental duplicate --image_folder ... from passthrough args."""
    out = []
    skip_next = False
    for a in extra:
        if skip_next:
            skip_next = False
            continue
        if a == "--image_folder":
            skip_next = True
            continue
        out.append(a)
    return out


def find_main_py(repo_root: Path) -> Path:
    """Find main.py in repo_root (direct), else recursively if unique."""
    direct = repo_root / "main.py"
    if direct.exists():
        return direct

    candidates = list(repo_root.rglob("main.py"))
    if len(candidates) == 1:
        return candidates[0]

    msg = f"Cannot find main.py at: {direct}\n"
    msg += f"Found candidates: {[str(c) for c in candidates]}\n"
    msg += "Please pass correct --repo_root (the folder that contains main.py)."
    raise FileNotFoundError(msg)


def split_wrapper_and_passthrough(argv: List[str], ap: argparse.ArgumentParser):
    """
    Only pass arguments AFTER `--` to main.py.
    Wrapper args are everything BEFORE `--`.
    """
    if "--" in argv:
        cut = argv.index("--")
        known_argv = argv[:cut]
        passthrough = argv[cut + 1:]
        args = ap.parse_args(known_argv)
        extra = passthrough
    else:
        # backward compatible: unknown args are forwarded (not recommended)
        args, extra = ap.parse_known_args(argv)
    return args, extra


def run_one_chunk(
    repo_root: Path,
    main_py: Path,
    chunk_imgs: List[Path],
    out_dir: Path,
    prefer_symlink: bool,
    python_bin: str,
    passthrough_args: List[str],
    make_viz: bool = False,
    samples_per_chunk: int = 12,
    echo_main_output: bool = True,
) -> ChunkResult:
    safe_mkdir(out_dir)
    chunk_img_dir = out_dir / "images"
    safe_mkdir(chunk_img_dir)

    # materialize images as symlinks/copies
    for p in chunk_imgs:
        link_or_copy(p, chunk_img_dir / p.name, prefer_symlink=prefer_symlink)

    cmd = [python_bin, str(main_py), "--image_folder", str(chunk_img_dir)] + passthrough_args

    log_path = out_dir / "run.log"
    with log_path.open("w", encoding="utf-8") as f:
        f.write("CMD: " + " ".join(cmd) + "\n\n")
        f.flush()

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"

    t0 = time.time()
    proc = subprocess.Popen(
        cmd,
        cwd=str(repo_root),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=env,
    )

    collected = []
    with log_path.open("a", encoding="utf-8") as f:
        assert proc.stdout is not None
        for line in proc.stdout:
            collected.append(line)
            f.write(line)
            f.flush()
            if echo_main_output:
                sys.stdout.write(line)
                sys.stdout.flush()

    proc.wait()
    secs = time.time() - t0
    returncode = int(proc.returncode)

    log_text = "".join(collected)
    tex = texture_stats(chunk_imgs)
    costs = parse_generic_costs(log_text)

    (out_dir / "stats.json").write_text(
        json.dumps(
            {
                "texture": tex,
                "costs": costs,
                "returncode": returncode,
                "seconds": secs,
                "cmd": cmd,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    if make_viz:
        try:
            save_input_viz(chunk_imgs, out_dir / "inputs_viz.png", n_samples=samples_per_chunk)
        except Exception as e:
            with log_path.open("a", encoding="utf-8") as f:
                f.write(f"\n[VIZ_ERROR] {repr(e)}\n")

    return ChunkResult(
        chunk_id=-1,
        start_idx=-1,
        end_idx=-1,
        n_images=len(chunk_imgs),
        out_dir=out_dir,
        ok=(returncode == 0),
        returncode=returncode,
        seconds=secs,
        tex=tex,
        costs=costs,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo_root", type=str, required=True, help="Path to VGGT-SLAM repo (contains main.py).")
    ap.add_argument("--image_folder", type=str, required=True, help="Folder containing your images.")
    ap.add_argument("--out_root", type=str, default="runs_chunked", help="Where to store outputs.")
    ap.add_argument("--chunk_size", type=int, default=100)
    ap.add_argument("--start_idx", type=int, default=0, help="Start index in sorted image list (inclusive).")
    ap.add_argument("--end_idx", type=int, default=-1, help="End index in sorted image list (inclusive). -1 means last.")
    ap.add_argument("--prefer_symlink", action="store_true", help="Use symlinks (fast, no duplication).")

    ap.add_argument("--viz", action="store_true", help="Save per-chunk input visualizations and summary plots.")
    ap.add_argument("--samples_per_chunk", type=int, default=12, help="How many sample images to visualize per chunk.")
    ap.add_argument("--no_echo", action="store_true", help="Do not echo main.py output to terminal (still written to run.log).")

    ap.add_argument("--python_bin", type=str, default=sys.executable, help="Python executable to run main.py.")

    args, extra = split_wrapper_and_passthrough(sys.argv[1:], ap)
    passthrough_args = strip_duplicate_image_folder_args(extra)

    repo_root = Path(args.repo_root).resolve()
    main_py = find_main_py(repo_root)

    img_folder = Path(args.image_folder).resolve()
    imgs_all = list_images(img_folder)
    if not imgs_all:
        raise RuntimeError(f"No images found in {img_folder} (extensions: {sorted(IMG_EXTS)})")

    start = max(0, int(args.start_idx))
    end = (len(imgs_all) - 1) if int(args.end_idx) < 0 else min(len(imgs_all) - 1, int(args.end_idx))
    if start > end:
        raise ValueError(f"start_idx({start}) > end_idx({end})")

    imgs = imgs_all[start:end + 1]
    chunk_size = max(1, int(args.chunk_size))

    out_root = Path(args.out_root).resolve()
    safe_mkdir(out_root)

    n = len(imgs)
    n_chunks = (n + chunk_size - 1) // chunk_size
    results: List[ChunkResult] = []

    print(f"[INFO] repo_root   : {repo_root}")
    print(f"[INFO] main.py     : {main_py}")
    print(f"[INFO] image_folder: {img_folder}")
    print(f"[INFO] images      : {len(imgs_all)} total, using [{start}:{end}] => {len(imgs)}")
    print(f"[INFO] chunk_size  : {chunk_size} => {n_chunks} chunks")
    print(f"[INFO] out_root    : {out_root}")
    print(f"[INFO] passthrough : {' '.join(passthrough_args) if passthrough_args else '(none)'}")
    print("")

    for ci in range(n_chunks):
        s = ci * chunk_size
        e = min(n, (ci + 1) * chunk_size) - 1
        chunk_imgs = imgs[s:e + 1]

        chunk_name = f"chunk_{ci:02d}_idx_{start + s:04d}-{start + e:04d}"
        out_dir = out_root / chunk_name

        print(f"\n========== RUN {ci+1}/{n_chunks}: {chunk_name} (n={len(chunk_imgs)}) ==========")
        r = run_one_chunk(
            repo_root=repo_root,
            main_py=main_py,
            chunk_imgs=chunk_imgs,
            out_dir=out_dir,
            prefer_symlink=bool(args.prefer_symlink),
            python_bin=args.python_bin,
            passthrough_args=passthrough_args,
            make_viz=bool(args.viz),
            samples_per_chunk=int(args.samples_per_chunk),
            echo_main_output=(not args.no_echo),
        )
        r.chunk_id = ci
        r.start_idx = start + s
        r.end_idx = start + e
        results.append(r)

        print(f"[CHUNK DONE] returncode={r.returncode}  seconds={r.seconds:.1f}  out_dir={r.out_dir}")

    summary_path = out_root / "chunk_summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "chunk_id", "start_idx", "end_idx", "n_images",
            "ok", "returncode", "seconds",
            "lap_var_mean", "grad_mean", "brightness_mean", "brightness_std", "blur_score",
            "cost_found", "cost_last", "cost_min", "cost_max", "cost_mean",
            "out_dir"
        ])
        for r in results:
            cost_found = r.costs.get("found", False)
            w.writerow([
                r.chunk_id, r.start_idx, r.end_idx, r.n_images,
                r.ok, r.returncode, f"{r.seconds:.3f}",
                r.tex.get("lap_var_mean", 0.0),
                r.tex.get("grad_mean", 0.0),
                r.tex.get("brightness_mean", 0.0),
                r.tex.get("brightness_std", 0.0),
                r.tex.get("blur_score", 0.0),
                cost_found,
                r.costs.get("last", ""),
                r.costs.get("min", ""),
                r.costs.get("max", ""),
                r.costs.get("mean", ""),
                str(r.out_dir),
            ])

    print(f"\n[DONE] Wrote: {summary_path}")
    print(f"       Logs per chunk : {out_root}/chunk_*/run.log")
    print(f"       Per-chunk stats: {out_root}/chunk_*/stats.json")
    if args.viz:
        save_summary_plots(summary_path, out_root)
        print(f"[VIZ] Wrote: {out_root / 'summary_texture_over_chunks.png'}")
        print(f"[VIZ] Wrote: {out_root / 'scatter_lap_vs_grad.png'}")
        print(f"[VIZ] Per-chunk: {out_root}/chunk_*/inputs_viz.png")


if __name__ == "__main__":
    main()

```

### setup.py

```python
from setuptools import setup, find_packages

setup(
    name='vggt_slam',
    version='0.1.0',
    description='A feedforward SLAM system optimized on the SL(4) manifold.',
    packages=find_packages(include=['evals', 'evals.*', 'vggt_slam', 'vggt_slam.*']),
)


```

### evals/__init__.py

```python

```

### evals/eval7_scenes_dense.py

```python
# adapted from https://github.com/rmurai0610/MASt3R-SLAM/tree/main
import argparse
import pathlib
import copy
import cv2
from termcolor import colored

import numpy as np
import open3d as o3d
from natsort import natsorted
from scipy.spatial.transform import Rotation

import evo
from evo.core import sync
import evo.core.metrics as metrics
from evo.tools import file_interface

import evals.geometry_eval_utils as geom_utils

def vggt_resize(img, depth, new_size = (392, 518)):
    resized_img = np.array(img)
    resized_depth = np.array(depth)
    H, W = img.shape[:2]

    new_H, new_W = new_size
    resized_img = cv2.resize(
        resized_img, (new_W, new_H), interpolation=cv2.INTER_LANCZOS4
    )
    resized_depth = cv2.resize(
        resized_depth, (new_W, new_H), interpolation=cv2.INTER_NEAREST
    )

    H1, W1 = resized_img.shape[:2]

    H2, W2 = resized_img.shape[:2]
    scale_w = W / W1
    scale_h = H / H1
    half_crop_w = (W1 - W2) / 2
    half_crop_h = (H1 - H2) / 2

    return np.ascontiguousarray(resized_img), np.ascontiguousarray(resized_depth), (scale_w, scale_h, half_crop_w, half_crop_h)

def load_7scenes(dataset, W, H, calib):
    """
    Returns the ground truth trajectory and point cloud in the world coordinate
    """
    subsample = 1  # TODO REMOVE THIS!
    rgb_files = natsorted(list((dataset / "seq-01").glob("*.color.png")))[::subsample]
    depth_files = natsorted(list((dataset / "seq-01").glob("*.depth.png")))[::subsample]
    pose_files = natsorted(list((dataset / "seq-01").glob("*.pose.txt")))[::subsample]
    fx, fy, cx, cy = calib  # kinect intrinsics
    # create rgbdimages
    rgbd_images = []
    gt_poses = []
    pcds = []
    gt_tum_traj_WC = []
    valid_masks = []

    for i, (rgb_file, depth_file, pose_file) in enumerate(zip(rgb_files, depth_files, pose_files)):
        color = cv2.imread(rgb_file.as_posix())
        color = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
        depth = cv2.imread(depth_file.as_posix(), cv2.IMREAD_UNCHANGED)
        pose_WC = np.loadtxt(pose_file)
        color, depth, resize_params = vggt_resize(color, depth)
        H1, W1 = color.shape[:2]
        fx1 = fx / resize_params[0]
        fy1 = fy / resize_params[1]
        cx1 = cx / resize_params[0] - resize_params[2]
        cy1 = cy / resize_params[1] - resize_params[3]

        # depth range of kinect is 0.5 - 4.5m
        depth[depth == 65535] = 0
        depth[depth > 4.5 * 1000] = 0
        depth = np.nan_to_num(depth, nan=0)
        valid_mask = depth > 0

        color = o3d.geometry.Image(color)
        depth = o3d.geometry.Image(depth)

        rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color,
            depth,
            depth_scale=1000,  # mm -> m
            depth_trunc=4.5,
            convert_rgb_to_intensity=False,
        )

        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
            rgbd_image,
            o3d.camera.PinholeCameraIntrinsic(
                width=W1, height=H1, fx=fx1, fy=fy1, cx=cx1, cy=cy1
            ),
            project_valid_depth_only=True,
        )
        pcd.transform(pose_WC)

        rgbd_images.append(rgbd_image)
        gt_poses.append(pose_WC)
        pcds.append(pcd)

        valid_masks.append(valid_mask)

        t_WC = pose_WC[:3, 3]
        R_WC = pose_WC[:3, :3]
        q_WC = Rotation.from_matrix(R_WC).as_quat()
        gt_tum_traj_WC.append(np.array([i * subsample, *t_WC, *q_WC]))

    return gt_tum_traj_WC, pcds, valid_masks


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="/home/<user>/Documents/MASt3R-SLAM/datasets/7-scenes/office")
    parser.add_argument("--gt", default="/home/<user>/Documents/MASt3R-SLAM/groundtruths/7-scenes/office.txt")
    parser.add_argument("--est", default="/home/<user>/Documents/vggt/office.txt")
    parser.add_argument("--no-viz", action="store_true")

    args = parser.parse_args()

    dataset = pathlib.Path(args.dataset)
    calib = 585.0, 585.0, 320.0, 240.0  # kinect intrinsics
    W, H = 640, 480
    K = np.array(
        [[calib[0], 0.0, calib[2]], [0.0, calib[1], calib[3]], [0.0, 0.0, 1.0]]
    )
    gt_traj, gt_pcds, valid_masks = load_7scenes(dataset, W, H, calib)

    # intrinsics = o3d.camera.PinholeCameraIntrinsic(W, H, K)
    if not args.no_viz:
        vis = o3d.visualization.Visualizer()
        vis.create_window()

    traj_ref = file_interface.read_tum_trajectory_file(args.gt)
    traj_est = file_interface.read_tum_trajectory_file(args.est)
    matches = sync.matching_time_indices(traj_ref.timestamps, traj_est.timestamps)

    traj_ref, traj_est = sync.associate_trajectories(traj_ref, traj_est)

    traj_est_aligned = copy.deepcopy(traj_est)
    r_a, t_a, s = traj_est_aligned.align(traj_ref, correct_scale=True, correct_only_scale=False)
    t_a = t_a.reshape(3, 1)

    # traj_est_aligned_poses = traj_est_aligned.poses_se3
    traj_est_poses = traj_est.poses_se3

    temp_path = args.est.replace(".txt", "_logs/")

    pcd_gt = []
    pcd_est_list = []
    for a, b in zip(matches[0], matches[1]):
        valid_mask = valid_masks[int(a)]
        gt_p = np.asarray(gt_pcds[int(a)].points)
        pcd_gt.append(gt_p)
        est_points = np.load(temp_path+str(a)+".0.npz")['pointcloud']
        est_mask = np.load(temp_path+str(a)+".0.npz")['mask']
        pcd_est_list.append(est_points[valid_mask & est_mask])

    pcd_est = np.concatenate([pcd for pcd in pcd_est_list], axis=0)
    pcd_est = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pcd_est))
    pcd_est.paint_uniform_color([0.0, 0.0, 1.0])

    center = np.asarray(pcd_est.get_center(), dtype=np.float64)
    # scale point cloud and given initial estimate of transformation.
    points = np.asarray(pcd_est.points)
    scaled_points = ((s*r_a) @ points.T + t_a).T
    pcd_est.points = o3d.utility.Vector3dVector(scaled_points)

    gt_pcd = np.concatenate(pcd_gt, axis=0)

    gt_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(gt_pcd))
    gt_pcd.paint_uniform_color([1.0, 0.0, 0.0])

    # print(f"Number of points in estimated point cloud: {len(pcd_est.points)}")
    # print(f"Number of points in ground truth point cloud: {len(gt_pcd.points)}")
    
    # Run ICP alignment.
    # Downsample both point clouds for ICP only.
    voxel_size = 0.05  # Adjust this depending on your scale and desired speed/accuracy tradeoff
    pcd_est_down = pcd_est.voxel_down_sample(voxel_size)
    gt_pcd_down = gt_pcd.voxel_down_sample(voxel_size)

    # Run ICP on downsampled point clouds
    reg_p2p = o3d.pipelines.registration.registration_icp(
        pcd_est_down,
        gt_pcd_down,
        max_correspondence_distance=0.1,  # Adjust to match voxel size
        init=np.eye(4),
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint()
    )

    # Apply the transformation to the full-resolution source point cloud
    transformation = reg_p2p.transformation
    pcd_est.transform(transformation)

    gt_mean = np.mean(np.asarray(gt_pcd.points), axis=0)
    est_mean = np.mean(np.asarray(pcd_est.points), axis=0)
    # print(f"Mean of the ground truth point cloud: {gt_mean}")
    # print(f"Mean of the estimated point cloud: {est_mean}")

    chamfer_dist, rmse_acc, rmse_comp, dists1, dists2 = (
        geom_utils.chamfer_distance_RMSE(gt_pcd, pcd_est, max_error=0.5)
    )

    # thresholds = np.arange(0, 1.0, 0.1)
    # valid1 = dists1.reshape(-1, 1) < thresholds.reshape(1, -1)
    # valid2 = dists2.reshape(-1, 1) < thresholds.reshape(1, -1)
    # print(valid1.sum(axis=0) / valid1.shape[0])
    # print(valid2.sum(axis=0) / valid2.shape[0])

    # print using colored
    print(colored("Dense eval results on", "green"), args.est)
    print(colored("RMSE acc:", "green"), rmse_acc)
    print(colored("RMSE comp:", "green"), rmse_comp)
    print(colored("Chamfer distance:", "green"), chamfer_dist)
    print()
    
    if not args.no_viz:
        vis.add_geometry(pcd_est, reset_bounding_box=False)
        vis.add_geometry(gt_pcd)
        vis.run()

```

### evals/geometry_eval_utils.py

```python
# adapted from https://github.com/rmurai0610/MASt3R-SLAM/tree/main
import pathlib
import lietorch
import torch
import numpy as np
import tqdm

# from mast3r_slam.lietorch_utils import as_SE3, lietorch_to_mat
from pykdtree.kdtree import KDTree as pyKDTree


def load_mast3r_slam(reconstruction_file, nanosec=False):
    reconstruction = torch.load(reconstruction_file)
    keyframes = {}
    est_traj = {}
    for keyframe_id, keyframe in reconstruction.items():
        timestamp = float(keyframe["timestamp"])
        if nanosec:
            timestamp /= 1e9
        T_WC, scale = lietorch_to_mat(keyframe["T_WC"], return_scale=True)
        # Apply scaling
        keyframes[timestamp] = {
            "T_WC": T_WC,
            "X": scale * np.array(keyframe["X_canon"], dtype=np.float64),
        }
        est_traj[timestamp] = as_SE3(keyframe["T_WC"]).data.numpy().reshape(-1).tolist()
    return keyframes, est_traj


def load_droid_slam(reconstruction_dir, nanosec=False):
    reconstruction_dir = pathlib.Path(reconstruction_dir)
    # load npy files
    disps = np.load(reconstruction_dir / "disps.npy")
    poses = np.load(reconstruction_dir / "poses.npy")
    timestamps = np.load(reconstruction_dir / "tstamps.npy")
    intrinsics = np.load(reconstruction_dir / "intrinsics.npy")
    keyframes = {}
    est_traj = {}
    for t, disp, pose, intrinsic in zip(timestamps, disps, poses, intrinsics):
        t = float(t)
        if nanosec:
            t /= 1e9

        pts = iproj(disp, intrinsic)
        T_WC = lietorch.SE3(torch.from_numpy(pose.reshape(1, -1)))
        keyframes[t] = {"T_WC": lietorch_to_mat(T_WC), "X": pts}
        est_traj[t] = T_WC.data.numpy().reshape(-1).tolist()
    return keyframes, est_traj


def find_visible_points(points, keyframes, W, H, calib):
    # check which points are visible from the keyframes
    fx, fy, cx, cy = calib
    points = torch.from_numpy(points).to(dtype=torch.float32, device="cuda")
    mask = torch.zeros(points.shape[0], dtype=bool, device="cuda")
    for keyframe in tqdm.tqdm(keyframes.values()):
        # if gt pose is not in the keyframe, it did not register with the ground truth
        if "gt_T_WC" not in keyframe:
            continue
        T_WC = keyframe["gt_T_WC"]
        R = T_WC[:3, :3].reshape(1, 3, 3)
        t = T_WC[:3, 3].reshape(1, 3, 1)
        Rinv = R.transpose(0, 2, 1)
        tinv = -Rinv @ t.reshape(1, 3, 1)
        Rinv = torch.from_numpy(Rinv).to(dtype=torch.float32, device="cuda")
        tinv = torch.from_numpy(tinv).to(dtype=torch.float32, device="cuda")
        point_C = (Rinv @ points.view(-1, 3, 1) + tinv).view(-1, 3)

        # apply projection
        z = point_C[:, 2]
        x = fx * point_C[:, 0] / z + cx
        y = fy * point_C[:, 1] / z + cy

        mask |= (y >= 0) & (y < H) & (x >= 0) & (x < W) & (z > 0)
    print(f"visible points: {mask.sum()} / {mask.shape[0]}")
    return points[mask].to(device="cpu", dtype=torch.float64).numpy()


def chamfer_distance(pcd_ref, pcd_est, max_error):
    ref, est = np.asarray(pcd_ref.points), np.asarray(pcd_est.points)
    kdtree_ref = pyKDTree(ref)
    kdtree_est = pyKDTree(est)
    dist1, _ = kdtree_ref.query(est)
    dist2, _ = kdtree_est.query(ref)
    dist1 = np.clip(dist1, 0, max_error)
    dist2 = np.clip(dist2, 0, max_error)
    chamfer_dist = 0.5 * np.mean(dist1) + 0.5 * np.mean(dist2)

    print("dist1", np.mean(dist1), np.median(dist1), np.max(dist1))
    print("dist2", np.mean(dist2), np.median(dist2), np.max(dist2))

    return chamfer_dist, dist1, dist2


def chamfer_distance_RMSE(pcd_ref, pcd_est, max_error):
    ref, est = np.asarray(pcd_ref.points), np.asarray(pcd_est.points)
    kdtree_ref = pyKDTree(ref)
    kdtree_est = pyKDTree(est)
    dist1, _ = kdtree_ref.query(est)
    dist2, _ = kdtree_est.query(ref)
    dist1 = np.clip(dist1, 0, max_error)
    dist2 = np.clip(dist2, 0, max_error)
    rmse_dist1 = np.sqrt(np.mean(dist1**2))
    rmse_dist2 = np.sqrt(np.mean(dist2**2))
    chamfer_dist = 0.5 * rmse_dist1 + 0.5 * rmse_dist2

    # print("dist1", rmse_dist1, np.mean(dist1), np.median(dist1), np.max(dist1))
    # print("dist2", rmse_dist2, np.mean(dist2), np.median(dist2), np.max(dist2))

    return chamfer_dist, rmse_dist1, rmse_dist2, dist1, dist2


def iproj(disps, intrinsics):
    """pinhole camera inverse projection"""
    ht, wd = disps.shape
    fx, fy, cx, cy = intrinsics
    x, y = np.meshgrid(np.arange(wd), np.arange(ht), indexing="xy")
    i = np.ones_like(disps)
    X = (x - cx) / fx
    Y = (y - cy) / fy
    pts = np.stack([X, Y, i], axis=-1)
    return pts * (1 / disps[..., None])
```

### evals/process_logs_7scenes.py

```python
import argparse
import pandas as pd
from pathlib import Path

parser = argparse.ArgumentParser(description="Process TUM results")
parser.add_argument("--submap_size", type=str, default="32", help="submap size to use")
args = parser.parse_args()

# Path to your 7-scenes results log
log_path = Path.cwd() / ("logs/" "7scenes_results_w" + args.submap_size + ".txt")

# Load the log file
df = pd.read_csv(log_path)

# Remove any average rows (we’ll recalculate)
df = df[df["Dataset"] != "Average"]

# Ensure correct data types
df["Run"] = df["Run"].astype(int)

print("=== Per-Experiment Results (Run x Dataset) ===")
for run in sorted(df["Run"].unique()):
    print(f"\n--- Run {run} ---")
    run_df = df[df["Run"] == run]
    for _, row in run_df.iterrows():
        scene = row["Dataset"]
        rmse = row["RMSE"]
        acc = row["RMSE acc"]
        comp = row["RMSE comp"]
        chamfer = row["Chamfer"]
        print(f"{scene}: RMSE={rmse:.4f}, Acc={acc:.4f}, Comp={comp:.4f}, Chamfer={chamfer:.4f}")

print("\n=== Per-Run Averages ===")
per_run_avg = df.groupby("Run")[["RMSE", "RMSE acc", "RMSE comp", "Chamfer"]].mean()
print(per_run_avg.round(4))

print("\n=== Per-Dataset Averages Across All Runs ===")
per_scene_avg = df.groupby("Dataset")[["RMSE", "RMSE acc", "RMSE comp", "Chamfer"]].mean()
print(per_scene_avg.round(4))

overall_avg = df[["RMSE", "RMSE acc", "RMSE comp", "Chamfer"]].mean()
print("\n=== Overall Averages Across All Runs and Scenes ===")
print(overall_avg.round(4))

```

### evals/process_logs_tum.py

```python
import argparse
import pandas as pd
from pathlib import Path

parser = argparse.ArgumentParser(description="Process TUM results")
parser.add_argument("--submap_size", type=str, default="32", help="submap size to use")
args = parser.parse_args()

# Path to your log file
log_path = Path.cwd() / ("logs/" "tum_results_w" + args.submap_size + ".txt")

# Load the CSV log
df = pd.read_csv(log_path)

# Clean: remove any existing 'Average' rows
df = df[df["Dataset"] != "Average"]

# Ensure proper data types
df["Run"] = df["Run"].astype(int)

print("=== Per-Experiment RMSE APE (Run x Dataset) ===")
for run in sorted(df["Run"].unique()):
    print(f"\n--- Run {run} ---")
    run_df = df[df["Run"] == run]
    for _, row in run_df.iterrows():
        print(f"{row['Dataset']}: {row['RMSE']:.4f}")

print("\n=== Per-Run Average RMSE APE ===")
per_run_avg = df.groupby("Run")["RMSE"].mean()
for run, val in per_run_avg.items():
    print(f"Run {run}: {val:.4f}")

print("\n=== Per-Dataset Average RMSE APE ===")
per_dataset_avg = df.groupby("Dataset")["RMSE"].mean()
for dataset, val in per_dataset_avg.items():
    print(f"{dataset}: {val:.4f}")

overall_avg = df["RMSE"].mean()
print("\n=== Overall Average RMSE APE Across All Runs ===")
print(f"Overall Average RMSE: {overall_avg:.4f}")

```

### scripts/align_points.py

```python
import open3d as o3d
import numpy as np

def preprocess_point_cloud(pcd, voxel_size):
    """Downsample and compute FPFH features."""
    pcd_down = pcd.voxel_down_sample(voxel_size)
    pcd_down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2, max_nn=30))
    fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down, o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 5, max_nn=100)
    )
    return pcd_down, fpfh

def visualize_alignment(source, target, transformation):
    """Visualize the aligned point clouds."""
    source_transformed = source.transform(transformation)

    source.paint_uniform_color([1, 0, 0])  # Red
    target.paint_uniform_color([0, 1, 0])  # Green

    o3d.visualization.draw_geometries([source_transformed, target], window_name="Aligned Point Clouds")

def register_point_clouds(source_np, target_np, voxel_size=0.05):
    """Find similarity transform (rotation, translation, and scale) between two point clouds."""

    source = o3d.geometry.PointCloud()
    source.points = o3d.utility.Vector3dVector(source_np)
    target = o3d.geometry.PointCloud()
    target.points = o3d.utility.Vector3dVector(target_np)

    # Preprocess
    src_down, src_fpfh = preprocess_point_cloud(source, voxel_size)
    tgt_down, tgt_fpfh = preprocess_point_cloud(target, voxel_size)

    # Global registration using RANSAC
    distance_threshold = voxel_size * 1.5
    result_ransac = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        src_down, tgt_down, src_fpfh, tgt_fpfh, True, distance_threshold,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(with_scaling=True),
        4,  # RANSAC iterations
        [o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
         o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold)],
        o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999)
    )

    # Refine with ICP
    result_icp = o3d.pipelines.registration.registration_icp(
        source, target, distance_threshold, result_ransac.transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(with_scaling=True)
    )

    # Extract transformation components
    transformation = result_icp.transformation
    R = transformation[:3, :3].copy()  # Make a writable copy
    t = transformation[:3, 3]          # Translation remains fine
    s = np.cbrt(np.linalg.det(R))      # Compute scale
    R /= s  # Normalize rotation to remove scale
    transformation = result_icp.transformation
    # visualize_alignment(source, target, transformation)

    return R, t, s, transformation
```

### scripts/eval_corridor_extra_metrics.py

```python
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import math
import argparse
from typing import Tuple, Optional, Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def read_poses_txt(path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Read poses.txt with format:
      frame_id x y z qx qy qz qw
    Returns:
      frame_ids: (N,)
      xyz: (N,3)
    """
    frame_ids = []
    xyz = []

    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 8:
                continue
            try:
                fid = float(parts[0])
                x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
            except Exception:
                continue
            frame_ids.append(fid)
            xyz.append([x, y, z])

    if len(xyz) == 0:
        return np.array([], dtype=float), np.zeros((0, 3), dtype=float)

    return np.array(frame_ids, dtype=float), np.asarray(xyz, dtype=float)


def pca_main_direction(xyz: np.ndarray) -> np.ndarray:
    """
    Compute main direction vector (unit) using PCA (SVD) on 3D positions.
    """
    if xyz.shape[0] < 2:
        return np.array([1.0, 0.0, 0.0], dtype=float)

    center = xyz.mean(axis=0, keepdims=True)
    X = xyz - center
    # SVD: X = U S Vt
    _, _, vt = np.linalg.svd(X, full_matrices=False)
    v = vt[0]
    v = v / (np.linalg.norm(v) + 1e-12)
    return v


def lateral_drift_stats(xyz: np.ndarray) -> Dict[str, float]:
    """
    Lateral drift = distance to main axis line (PCA first component).
    For each point p:
      d = || (p - p0) - dot(p - p0, v)*v ||
    Returns rmse/p95/max (meters) + path length etc.
    """
    if xyz.shape[0] < 3:
        return {
            "lateral_rmse_m": float("nan"),
            "lateral_p95_m": float("nan"),
            "lateral_max_m": float("nan"),
            "along_length_m": float("nan"),
        }

    v = pca_main_direction(xyz)
    p0 = xyz[0]
    dp = xyz - p0[None, :]

    # along-track scalar
    s = dp @ v
    # perpendicular component
    perp = dp - s[:, None] * v[None, :]
    lateral = np.linalg.norm(perp, axis=1)

    lateral_rmse = float(np.sqrt(np.mean(lateral ** 2)))
    lateral_p95 = float(np.percentile(lateral, 95))
    lateral_max = float(np.max(lateral))
    along_len = float(np.max(s) - np.min(s))  # extent along main direction

    return {
        "lateral_rmse_m": lateral_rmse,
        "lateral_p95_m": lateral_p95,
        "lateral_max_m": lateral_max,
        "along_length_m": along_len,
    }


def make_plots(xyz: np.ndarray, out_dir: str, prefix: str = "") -> None:
    """
    Save:
      - lateral_profile.png: along-track s vs lateral distance
      - pca_topdown.png: trajectory projected to (PC1, PC2) plane
    """
    if xyz.shape[0] < 3:
        return

    os.makedirs(out_dir, exist_ok=True)

    # PCA basis
    center = xyz.mean(axis=0, keepdims=True)
    X = xyz - center
    _, _, vt = np.linalg.svd(X, full_matrices=False)
    pc1 = vt[0] / (np.linalg.norm(vt[0]) + 1e-12)
    pc2 = vt[1] / (np.linalg.norm(vt[1]) + 1e-12)

    p0 = xyz[0]
    dp = xyz - p0[None, :]

    s = dp @ pc1
    perp = dp - s[:, None] * pc1[None, :]
    lateral = np.linalg.norm(perp, axis=1)

    # Figure 1: lateral profile
    plt.figure(figsize=(8, 4))
    plt.plot(s, lateral, marker="o", linewidth=1)
    plt.xlabel("Along-track (m)  [projection onto PC1]")
    plt.ylabel("Lateral distance (m)  [distance to main axis]")
    plt.title("Lateral drift profile")
    plt.grid(True, alpha=0.3)
    fn1 = os.path.join(out_dir, f"{prefix}lateral_profile.png")
    plt.tight_layout()
    plt.savefig(fn1, dpi=160)
    plt.close()

    # Figure 2: topdown in PCA plane
    x1 = (xyz - center) @ pc1
    x2 = (xyz - center) @ pc2
    plt.figure(figsize=(5, 5))
    plt.plot(x1, x2, marker="o", linewidth=1)
    plt.xlabel("PC1 (m)")
    plt.ylabel("PC2 (m)")
    plt.title("Trajectory in PCA plane (top-down)")
    plt.axis("equal")
    plt.grid(True, alpha=0.3)
    fn2 = os.path.join(out_dir, f"{prefix}pca_topdown.png")
    plt.tight_layout()
    plt.savefig(fn2, dpi=160)
    plt.close()


def eval_single(poses_file: str, out_dir: Optional[str], write_plots: bool) -> Dict[str, float]:
    fids, xyz = read_poses_txt(poses_file)
    stats = lateral_drift_stats(xyz)
    if write_plots and out_dir:
        make_plots(xyz, out_dir=out_dir, prefix="")
    return stats


def update_metrics_csv(metrics_csv: str, out_csv: str, plot: bool) -> None:
    df = pd.read_csv(metrics_csv)

    # ensure columns
    for c in ["lateral_rmse_m", "lateral_p95_m", "lateral_max_m", "along_length_m"]:
        if c not in df.columns:
            df[c] = np.nan

    n = len(df)
    for i in range(n):
        poses_file = str(df.loc[i, "poses_file"]) if "poses_file" in df.columns else ""
        run_dir = str(df.loc[i, "run_dir"]) if "run_dir" in df.columns else ""

        if not poses_file or not os.path.isfile(poses_file):
            continue

        try:
            stats = eval_single(
                poses_file=poses_file,
                out_dir=run_dir if (plot and run_dir) else None,
                write_plots=plot
            )
            for k, v in stats.items():
                df.loc[i, k] = v
        except Exception as e:
            print(f"[WARN] failed on row {i} poses={poses_file}: {e}")
            continue

    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"[DONE] wrote: {out_csv}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--poses", type=str, default="", help="single poses.txt to evaluate")
    ap.add_argument("--metrics_csv", type=str, default="", help="batch update: corridor_compare_metrics.csv")
    ap.add_argument("--out_csv", type=str, default="", help="output csv (for batch mode)")
    ap.add_argument("--plot", action="store_true", help="save plots into run_dir (batch) or out_dir (single)")
    ap.add_argument("--out_dir", type=str, default="", help="single mode: output dir for plots")
    args = ap.parse_args()

    if args.poses:
        out_dir = args.out_dir if args.out_dir else (os.path.dirname(args.poses) or ".")
        stats = eval_single(args.poses, out_dir=out_dir, write_plots=args.plot)
        print("[LATERAL]")
        for k, v in stats.items():
            print(f"  {k}: {v}")
        if args.plot:
            print(f"[PLOTS] saved into: {out_dir}")
        return

    if args.metrics_csv:
        out_csv = args.out_csv if args.out_csv else args.metrics_csv.replace(".csv", "_plus.csv")
        update_metrics_csv(metrics_csv=args.metrics_csv, out_csv=out_csv, plot=args.plot)
        return

    raise SystemExit("Provide --poses or --metrics_csv")


if __name__ == "__main__":
    main()

```

### scripts/ros_to_jpg.py

```python
import rosbag
import rospy
import numpy as np
import csv
import os
import cv2
import gtsam
import json
from cv_bridge import CvBridge

######## user params ##########
RGB_TOPIC_NAME = "/device_0/sensor_1/Color_0/image/data"


SAVE_LOC = "/home/<user>/cam_ws/imgs"
BAG_LOC = "/home/<user>/Documents/"
bag = "20250417_141216"


downsample = 8


data_bag = rosbag.Bag(BAG_LOC+bag+".bag")

# make folder structure
os.makedirs(SAVE_LOC + "/" + bag)
os.makedirs(SAVE_LOC + "/" + bag + "/images")


count = 0
index = 0

timestamps = []

for topic, msg, t in data_bag.read_messages(topics=[RGB_TOPIC_NAME]):
    if topic == RGB_TOPIC_NAME:
        count += 1
        if count%downsample==0: 
            timestamps.append(msg.header.stamp)
            bridge = CvBridge()
            cv_image = bridge.imgmsg_to_cv2(msg)
            cv_image.astype(np.uint8)
            cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
            cv2.imwrite(SAVE_LOC + "/" + bag + "/images/" + "rgb_" + str(index) + '.jpg', cv_image)
            

            index += 1

cv2.destroyAllWindows()


```

### scripts/run_corridor_compare.py

```python
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse, csv, os, sys, time
from datetime import datetime
from pathlib import Path
import subprocess
import numpy as np
import re


def ts() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def stream(cmd: list[str], cwd: Path, log_path: Path) -> int:
    ensure_dir(log_path.parent)
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    with log_path.open("w", encoding="utf-8") as f:
        f.write(f"[CMD] {' '.join(cmd)}\n")
        f.write(f"[CWD] {cwd}\n")
        f.write(f"[TIME] {datetime.now().isoformat()}\n")
        f.write("-" * 80 + "\n")
        f.flush()

        proc = subprocess.Popen(
            cmd,
            cwd=str(cwd),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True,
            env=env,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            f.write(line)
            f.flush()
        return proc.wait()


def parse_poses_txt(p: Path):
    if not p.exists():
        return None
    rows = []
    for ln in p.read_text(encoding="utf-8", errors="ignore").splitlines():
        s = ln.strip()
        if not s or s.startswith("#"):
            continue
        parts = re.split(r"[,\s]+", s)
        try:
            vals = [float(x) for x in parts]
        except Exception:
            continue
        rows.append(vals)
    if len(rows) < 2:
        return None
    arr = np.array(rows, dtype=np.float64)
    # frame_id x y z qx qy qz qw
    if arr.shape[1] >= 4:
        return arr[:, 1:4]
    return None


def metrics_from_xyz(xyz: np.ndarray) -> dict:
    N = xyz.shape[0]
    if N < 2:
        return {"n_poses": int(N)}

    dp = xyz[1:] - xyz[:-1]
    step = np.linalg.norm(dp, axis=1)
    path_len = float(step.sum())
    loop_err = float(np.linalg.norm(xyz[-1] - xyz[0]))
    drift_ratio = float(loop_err / (path_len + 1e-12))

    # ---- Corridor-sensitive: lateral drift via PCA line fit in 3D ----
    p0 = xyz[0]
    X = xyz - p0
    # PCA: first principal direction
    # SVD of centered points
    Xc = X - X.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(Xc, full_matrices=False)
    v = vt[0]
    nv = np.linalg.norm(v)
    if not np.isfinite(nv) or nv < 1e-12:
        v = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    else:
        v = v / nv

    t = (xyz - p0) @ v
    proj = p0[None, :] + t[:, None] * v[None, :]
    lateral = np.linalg.norm(xyz - proj, axis=1)
    lateral_rmse = float(np.sqrt(np.mean(lateral**2)))
    lateral_p95 = float(np.percentile(lateral, 95))
    lateral_max = float(np.max(lateral))
    along_length = float(np.max(t) - np.min(t))

    return {
        "n_poses": int(N),
        "path_length_m": path_len,
        "loop_trans_err_m": loop_err,
        "drift_ratio": drift_ratio,
        "step_trans_mean_m": float(step.mean()),
        "step_trans_p95_m": float(np.percentile(step, 95)),
        "step_trans_max_m": float(step.max()),
        "lateral_rmse_m": lateral_rmse,
        "lateral_p95_m": lateral_p95,
        "lateral_max_m": lateral_max,
        "along_length_m": along_length,
    }


def append_csv(csv_path: Path, row: dict, fieldnames: list[str]) -> None:
    ensure_dir(csv_path.parent)
    write_header = not csv_path.exists()
    with csv_path.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in fieldnames})


def _strip_flag(args: list[str], flag: str, has_value: bool = False) -> list[str]:
    """Remove all occurrences of a flag (optionally with its value)."""
    out = []
    i = 0
    while i < len(args):
        if args[i] == flag:
            if has_value and i + 1 < len(args):
                i += 2
            else:
                i += 1
            continue
        out.append(args[i])
        i += 1
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", type=str, required=True)
    ap.add_argument("--data", type=str, default="DATA/corridor_0300-0399")
    ap.add_argument("--max_loops", type=int, default=1)
    ap.add_argument("--repo_root", type=str, default=".")
    ap.add_argument("--python", type=str, default=sys.executable)
    ap.add_argument("--main_py", type=str, default="main.py")
    ap.add_argument("--metrics_csv", type=str, default="LOG/corridor_compare_metrics.csv")
    ap.add_argument(
        "--modes",
        nargs="+",
        default=["base", "sem_gate", "sem_weight", "both"],
        choices=["base", "sem_gate", "sem_weight", "both", "baseline", "semantic"],
        help="Ablations: base(no semantic), sem_gate(only gate), sem_weight(only weight), both(gate+weight). "
             "Aliases: baseline->base, semantic->both",
    )
    ap.add_argument("passthrough", nargs=argparse.REMAINDER)
    args = ap.parse_args()

    repo = Path(args.repo_root).resolve()
    data = (repo / args.data).resolve()
    main_py = (repo / args.main_py).resolve()
    log_dir = repo / "LOG"
    run_root = repo / "RUNS" / args.tag
    ensure_dir(log_dir)
    ensure_dir(run_root)

    fieldnames = [
        "time", "tag", "mode", "returncode", "seconds",
        "dataset", "run_dir", "log_file", "poses_file",
        "n_poses", "path_length_m", "loop_trans_err_m", "drift_ratio",
        "step_trans_mean_m", "step_trans_p95_m", "step_trans_max_m",
        "cmd",
        "lateral_rmse_m", "lateral_p95_m", "lateral_max_m", "along_length_m",
    ]

    passthrough = args.passthrough
    if passthrough and passthrough[0] == "--":
        passthrough = passthrough[1:]

    # Runner is authoritative for semantic backend enable; prevent accidental contamination.
    passthrough = _strip_flag(passthrough, "--use_semantic_backend", has_value=False)

    # You can still pass these, but runner will override per-mode by appending force_args at the end.
    # (Argparse takes the last occurrence.)

    modes = []
    for m in args.modes:
        if m == "baseline":
            modes.append("base")
        elif m == "semantic":
            modes.append("both")
        else:
            modes.append(m)

    for mode in modes:
        stamp = ts()
        run_dir = run_root / mode / stamp
        ensure_dir(run_dir)

        log_file = log_dir / f"{args.tag}__{mode}__{stamp}.log"
        poses_file = run_dir / "poses.txt"
        edge_stats_file = run_dir / "edge_stats.csv"

        cmd = [
            args.python, str(main_py),
            "--image_folder", str(data),
            "--max_loops", str(args.max_loops),
            "--log_results",
            "--log_path", str(poses_file),
            "--edge_stats_path", str(edge_stats_file),
        ]

        # semantic backend toggle by mode
        if mode in ("sem_gate", "sem_weight", "both"):
            cmd += ["--use_semantic_backend"]

        # common passthrough (sanitized)
        cmd += list(passthrough)

        # force args per ablation (placed last => wins)
        force_args: list[str] = []
        if mode == "base":
            force_args += ["--semantic_weight_mode", "off", "--disable_semantic_gate"]
        elif mode == "sem_gate":
            force_args += ["--semantic_weight_mode", "off"]   # gate only
        elif mode == "sem_weight":
            force_args += ["--disable_semantic_gate"]         # weight only (keep your weight_mode in passthrough)
        elif mode == "both":
            pass

        cmd += force_args

        print("\n" + "=" * 80)
        print(f"[RUN] tag={args.tag} mode={mode}")
        print(f"[DATA] {data}")
        print(f"[RUN_DIR] {run_dir}")
        print(f"[LOG] {log_file}")
        print(f"[CMD] {' '.join(cmd)}")
        print("=" * 80 + "\n")

        t0 = time.time()
        rc = stream(cmd, cwd=repo, log_path=log_file)
        sec = time.time() - t0

        xyz = parse_poses_txt(poses_file) if rc == 0 else None
        met = {}
        if xyz is not None:
            met = metrics_from_xyz(xyz)

        row = {
            "time": now_iso(),
            "tag": args.tag,
            "mode": mode,
            "returncode": rc,
            "seconds": f"{sec:.3f}",
            "dataset": str(data),
            "run_dir": str(run_dir),
            "log_file": str(log_file),
            "poses_file": str(poses_file) if poses_file.exists() else "",
            "cmd": " ".join(cmd),
        }
        row.update(met)
        append_csv((repo / args.metrics_csv).resolve(), row, fieldnames)

        if rc != 0:
            print(f"[WARN] run failed (rc={rc}). see log: {log_file}")
        elif xyz is None:
            print(f"[WARN] poses exists but not parsed: {poses_file}")

    print(f"\n[DONE] metrics -> {repo / args.metrics_csv}")
    print(f"[DONE] runs    -> {run_root}")
    print(f"[DONE] logs    -> {log_dir}")


if __name__ == "__main__":
    main()

```

### scripts/run_corridor_compare_plus.py

```python
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, csv, os, re, shlex, sys, time, subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple, Dict
import numpy as np

# ---------------------------
# Utils
# ---------------------------
def ts() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")

def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def repo_root_from_script() -> Path:
    # scripts/xxx.py -> repo root
    return Path(__file__).resolve().parents[1]

def safe_float(x: str):
    try:
        return float(x)
    except Exception:
        return None

def stream_subprocess(cmd, cwd: Path, log_path: Path, env: Dict[str, str]) -> int:
    ensure_dir(log_path.parent)
    with log_path.open("w", encoding="utf-8") as f:
        f.write(f"[CMD] {' '.join(cmd)}\n")
        f.write(f"[CWD] {cwd}\n")
        f.write(f"[TIME] {datetime.now().isoformat()}\n")
        f.write("-"*80 + "\n")
        f.flush()

        proc = subprocess.Popen(
            cmd, cwd=str(cwd),
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1, universal_newlines=True, env=env
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            sys.stdout.write(line); sys.stdout.flush()
            f.write(line); f.flush()
        return proc.wait()

# ---------------------------
# Trajectory parsing + metrics
# poses.txt format (from map.write_poses_to_file):
# frame_id x y z qx qy qz qw
# ---------------------------
@dataclass
class Traj:
    t: np.ndarray
    p: np.ndarray
    q: Optional[np.ndarray] = None  # xyzw

def quat_normalize(q: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(q, axis=-1, keepdims=True) + 1e-12
    return q / n

def quat_conj(q: np.ndarray) -> np.ndarray:
    qc = q.copy()
    qc[..., :3] *= -1.0
    return qc

def quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    ax, ay, az, aw = a[...,0], a[...,1], a[...,2], a[...,3]
    bx, by, bz, bw = b[...,0], b[...,1], b[...,2], b[...,3]
    x = aw*bx + ax*bw + ay*bz - az*by
    y = aw*by - ax*bz + ay*bw + az*bx
    z = aw*bz + ax*by - ay*bx + az*bw
    w = aw*bw - ax*bx - ay*by - az*bz
    return np.stack([x,y,z,w], axis=-1)

def quat_to_rot_angle_deg(q_rel: np.ndarray) -> np.ndarray:
    q_rel = quat_normalize(q_rel)
    w = np.clip(q_rel[...,3], -1.0, 1.0)
    ang = 2.0 * np.arccos(w)
    return np.degrees(ang)

def parse_poses_txt(path: Path) -> Optional[Traj]:
    if not path.exists():
        return None
    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    rows = []
    for ln in lines:
        s = ln.strip()
        if not s or s.startswith("#"):
            continue
        parts = re.split(r"[,\s]+", s)
        vals = [safe_float(x) for x in parts]
        if any(v is None for v in vals):
            continue
        rows.append([float(v) for v in vals])
    if len(rows) < 2:
        return None
    arr = np.array(rows, dtype=np.float64)
    if arr.shape[1] >= 8:
        t = arr[:,0]
        p = arr[:,1:4]
        q = arr[:,4:8]  # xyzw
        return Traj(t=t, p=p, q=q)
    if arr.shape[1] == 4:
        return Traj(t=arr[:,0], p=arr[:,1:4], q=None)
    return None

def compute_metrics(traj: Traj, jump_trans_thresh: float = 0.5, jump_rot_thresh_deg: float = 30.0) -> Dict[str, object]:
    p = traj.p
    N = p.shape[0]
    out = {
        "n_poses": int(N),
        "path_length_m": "",
        "loop_trans_err_m": "",
        "loop_rot_err_deg": "",
        "drift_ratio": "",
        "step_trans_mean_m": "",
        "step_trans_p95_m": "",
        "step_trans_max_m": "",
        "step_rot_mean_deg": "",
        "step_rot_p95_deg": "",
        "step_rot_max_deg": "",
        "jump_trans_count": "",
        "jump_rot_count": "",
    }
    if N < 2:
        return out

    dp = p[1:] - p[:-1]
    step_trans = np.linalg.norm(dp, axis=1)
    path_len = float(step_trans.sum())
    loop_trans = float(np.linalg.norm(p[-1] - p[0]))

    out["path_length_m"] = path_len
    out["loop_trans_err_m"] = loop_trans
    out["drift_ratio"] = (loop_trans / (path_len + 1e-12)) if path_len > 0 else ""

    out["step_trans_mean_m"] = float(np.mean(step_trans))
    out["step_trans_p95_m"]  = float(np.percentile(step_trans, 95))
    out["step_trans_max_m"]  = float(np.max(step_trans))
    out["jump_trans_count"]  = int(np.sum(step_trans > jump_trans_thresh))

    if traj.q is not None and traj.q.shape[0] == N:
        q = quat_normalize(traj.q)
        q_rel = quat_mul(q[1:], quat_conj(q[:-1]))
        step_rot = quat_to_rot_angle_deg(q_rel)
        q_loop = quat_mul(q[-1], quat_conj(q[0]))
        out["loop_rot_err_deg"] = float(quat_to_rot_angle_deg(q_loop)[()])

        out["step_rot_mean_deg"] = float(np.mean(step_rot))
        out["step_rot_p95_deg"]  = float(np.percentile(step_rot, 95))
        out["step_rot_max_deg"]  = float(np.max(step_rot))
        out["jump_rot_count"]    = int(np.sum(step_rot > jump_rot_thresh_deg))

    return out

def parse_submaps_and_loops_from_log(log_file: Path) -> Tuple[str, str]:
    """
    Parse:
      Total number of submaps in map X
      Total number of loop closures in map Y
    """
    if not log_file.exists():
        return "", ""
    txt = log_file.read_text(encoding="utf-8", errors="ignore")
    m1 = re.findall(r"Total number of submaps in map\s+(\d+)", txt)
    m2 = re.findall(r"Total number of loop closures in map\s+(\d+)", txt)
    n_submaps = m1[-1] if m1 else ""
    n_loops   = m2[-1] if m2 else ""
    return n_submaps, n_loops

def append_csv(csv_path: Path, row: Dict[str, object], fieldnames):
    ensure_dir(csv_path.parent)
    write_header = not csv_path.exists()
    with csv_path.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in fieldnames})

# ---------------------------
# Main runner
# ---------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo_root", type=str, default="", help="VGGT-SLAM repo root (default: auto)")
    ap.add_argument("--main_py", type=str, default="main.py")
    ap.add_argument("--dataset", type=str, default="corridor_0300-0399", help="Only run this dataset folder under DATA/")
    ap.add_argument("--tag", type=str, default="corridor_sem_vs_base", help="RUNS/<tag>/..., LOG/<tag>__*.log")
    ap.add_argument("--max_loops", type=int, default=1)
    ap.add_argument("--semantic_min_sim", type=float, default=None, help="if provided, pass --semantic_min_sim X to main.py")
    ap.add_argument("--disable_loop_closure", action="store_true", help="pass --disable_loop_closure to main.py")
    ap.add_argument("--jump_trans_thresh", type=float, default=0.5)
    ap.add_argument("--jump_rot_thresh_deg", type=float, default=30.0)
    ap.add_argument("--metrics_csv", type=str, default="LOG/corridor_compare_metrics.csv")

    # pass-through for extra main.py args: use "-- <args...>"
    ap.add_argument("passthrough", nargs=argparse.REMAINDER)
    args = ap.parse_args()

    repo_root = Path(args.repo_root).resolve() if args.repo_root else repo_root_from_script()
    main_py = (repo_root / args.main_py).resolve()
    if not main_py.exists():
        print(f"[ERROR] main.py not found: {main_py}", file=sys.stderr)
        sys.exit(2)

    data_dir = (repo_root / "DATA" / args.dataset).resolve()
    if not data_dir.exists():
        print(f"[ERROR] dataset not found: {data_dir}", file=sys.stderr)
        sys.exit(2)

    log_dir = (repo_root / "LOG").resolve(); ensure_dir(log_dir)
    run_root = (repo_root / "RUNS" / args.tag).resolve(); ensure_dir(run_root)
    metrics_csv = (repo_root / args.metrics_csv).resolve()

    # count images (rough)
    n_images = len([p for p in data_dir.glob("*") if p.is_file() and p.suffix.lower() in [".jpg",".jpeg",".png",".bmp",".webp"]])

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"

    passthrough = args.passthrough
    if passthrough and passthrough[0] == "--":
        passthrough = passthrough[1:]

    # two modes only
    modes = [
        ("baseline", []),
        ("semantic", ["--use_semantic_backend"]),
    ]

    common_extra = ["--max_loops", str(args.max_loops)]
    if args.semantic_min_sim is not None:
        common_extra += ["--semantic_min_sim", str(args.semantic_min_sim)]
    if args.disable_loop_closure:
        common_extra += ["--disable_loop_closure"]

    fieldnames = [
        "time","tag","mode","returncode","seconds",
        "dataset","n_images",
        "run_dir","log_file","poses_file",
        "n_submaps","n_loops",
        "n_poses","path_length_m","loop_trans_err_m","loop_rot_err_deg","drift_ratio",
        "step_trans_mean_m","step_trans_p95_m","step_trans_max_m",
        "step_rot_mean_deg","step_rot_p95_deg","step_rot_max_deg",
        "jump_trans_count","jump_rot_count",
        "cmd",
    ]

    results = {}

    for mode, mode_args in modes:
        run_stamp = ts()
        run_dir = run_root / mode / run_stamp
        ensure_dir(run_dir)
        poses_file = run_dir / "poses.txt"
        log_file = log_dir / f"{args.tag}__{mode}__{run_stamp}.log"

        cmd = [sys.executable, str(main_py), "--image_folder", str(data_dir)]
        cmd += common_extra
        cmd += mode_args
        cmd += ["--log_results", "--log_path", str(poses_file)]
        cmd += passthrough

        print("\n" + "="*80)
        print(f"[RUN] tag={args.tag} mode={mode}")
        print(f"[DATA] {data_dir}")
        print(f"[RUN_DIR] {run_dir}")
        print(f"[LOG] {log_file}")
        print(f"[CMD] {' '.join(cmd)}")
        print("="*80 + "\n")

        t0 = time.time()
        rc = stream_subprocess(cmd, cwd=repo_root, log_path=log_file, env=env)
        sec = time.time() - t0

        traj = parse_poses_txt(poses_file) if rc == 0 else None
        met = compute_metrics(traj, args.jump_trans_thresh, args.jump_rot_thresh_deg) if traj else {k:"" for k in [
            "n_poses","path_length_m","loop_trans_err_m","loop_rot_err_deg","drift_ratio",
            "step_trans_mean_m","step_trans_p95_m","step_trans_max_m",
            "step_rot_mean_deg","step_rot_p95_deg","step_rot_max_deg",
            "jump_trans_count","jump_rot_count",
        ]}
        n_submaps, n_loops = parse_submaps_and_loops_from_log(log_file)

        row = {
            "time": now_iso(),
            "tag": args.tag,
            "mode": mode,
            "returncode": rc,
            "seconds": f"{sec:.3f}",
            "dataset": str(data_dir),
            "n_images": n_images,
            "run_dir": str(run_dir),
            "log_file": str(log_file),
            "poses_file": str(poses_file),
            "n_submaps": n_submaps,
            "n_loops": n_loops,
            "cmd": " ".join(cmd),
        }
        row.update(met)
        append_csv(metrics_csv, row, fieldnames)

        results[mode] = row

        if rc != 0:
            print(f"[WARN] run failed (rc={rc}). see log: {log_file}")
        elif traj is None:
            print(f"[WARN] poses.txt not parsed or too short -> {poses_file}")

    # print delta (semantic - baseline)
    if "baseline" in results and "semantic" in results:
        b = results["baseline"]
        s = results["semantic"]
        keys = ["loop_trans_err_m","drift_ratio","path_length_m","seconds","n_loops","n_poses"]
        print("\n=== semantic - baseline (delta) ===")
        for k in keys:
            try:
                fb = float(b[k])
                fs = float(s[k])
                print(f"{k:16s}: {fs - fb:+.6f}")
            except Exception:
                print(f"{k:16s}: (skip) baseline={b.get(k,'')} semantic={s.get(k,'')}")
        print(f"\n[DONE] metrics -> {metrics_csv}")
        print(f"[DONE] runs    -> {run_root}")
        print(f"[DONE] logs    -> {log_dir}")

if __name__ == "__main__":
    main()

```

### scripts/run_semantic_suite.py

```python
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Run baseline vs semantic (or any two modes) on selected datasets, and ALWAYS save logs into <repo>/LOG/.

Typical usage:
  python scripts/run_semantic_suite.py \
    --data_root DATA \
    --datasets control_0100-0199 corridor_0300-0399 \
    --modes baseline semantic \
    --baseline_args "--max_loops 1" \
    --semantic_args "--max_loops 1 --use_semantic_backend" \
    --tag sem_v1

Notes:
- This script streams main.py stdout/stderr to terminal AND to a timestamped log file under LOG/.
- It does NOT assume main.py supports an output-dir flag. If you later add one, just include it in *_args.
"""
from __future__ import annotations

import argparse
import csv
import os
import shlex
import sys
import time
from datetime import datetime
from pathlib import Path
import subprocess


def repo_root_from_script() -> Path:
    # scripts/run_semantic_suite.py -> repo root = parent of scripts/
    return Path(__file__).resolve().parents[1]


def ts() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def stream_subprocess(cmd: list[str], cwd: Path, log_path: Path, env: dict[str, str]) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)

    with log_path.open("w", encoding="utf-8") as f:
        f.write(f"[CMD] {' '.join(cmd)}\n")
        f.write(f"[CWD] {cwd}\n")
        f.write(f"[TIME] {datetime.now().isoformat()}\n")
        f.write("-" * 80 + "\n")
        f.flush()

        proc = subprocess.Popen(
            cmd,
            cwd=str(cwd),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True,
            env=env,
        )

        assert proc.stdout is not None
        for line in proc.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            f.write(line)
            f.flush()

        return proc.wait()


def append_summary(csv_path: Path, row: dict) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not csv_path.exists()

    fieldnames = [
        "time",
        "tag",
        "dataset",
        "mode",
        "image_folder",
        "returncode",
        "seconds",
        "log_file",
        "cmd",
    ]

    with csv_path.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in fieldnames})


def main():
    parser = argparse.ArgumentParser(description="Run main.py on datasets with baseline/semantic modes and log to <repo>/LOG/")
    parser.add_argument("--repo_root", type=str, default="", help="VGGT-SLAM repo root. Default: auto-detect from this script.")
    parser.add_argument("--main_py", type=str, default="main.py", help="Path to main.py (relative to repo_root).")

    parser.add_argument("--data_root", type=str, default="DATA", help="Data root under repo.")
    parser.add_argument("--datasets", nargs="+", default=["control_0100-0199", "corridor_0300-0399"], help="Dataset folder names under data_root.")

    parser.add_argument("--modes", nargs="+", default=["baseline", "semantic"], help="Run modes (e.g., baseline semantic).")

    parser.add_argument("--baseline_args", type=str, default="", help='Extra args string for baseline, e.g. "--max_loops 1"')
    parser.add_argument("--semantic_args", type=str, default="", help='Extra args string for semantic, e.g. "--max_loops 1 --use_semantic_backend"')

    parser.add_argument("--tag", type=str, default="", help="Experiment tag for log/summary naming.")
    parser.add_argument("--log_dir", type=str, default="LOG", help="Log directory under repo.")
    parser.add_argument("--summary_csv", type=str, default="LOG/semantic_suite_summary.csv", help="Summary CSV path under repo.")
    parser.add_argument("--dry_run", action="store_true", help="Only print commands, do not execute.")

    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve() if args.repo_root else repo_root_from_script()
    main_py = (repo_root / args.main_py).resolve()
    if not main_py.exists():
        print(f"[ERROR] main.py not found: {main_py}", file=sys.stderr)
        sys.exit(2)

    log_dir = (repo_root / args.log_dir).resolve()
    log_dir.mkdir(parents=True, exist_ok=True)

    summary_csv = (repo_root / args.summary_csv).resolve()

    data_root = (repo_root / args.data_root).resolve()
    if not data_root.exists():
        print(f"[ERROR] data_root not found: {data_root}", file=sys.stderr)
        sys.exit(2)

    mode_to_args = {
        "baseline": args.baseline_args,
        "semantic": args.semantic_args,
    }

    # Ensure real-time flush from python + downstream
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"

    for ds in args.datasets:
        image_folder = data_root / ds
        if not image_folder.exists():
            print(f"[WARN] dataset not found, skip: {image_folder}")
            continue

        for mode in args.modes:
            extra = mode_to_args.get(mode, "")
            extra_list = shlex.split(extra)

            tag = args.tag.strip() or "no_tag"
            log_name = f"{tag}__{ds}__{mode}__{ts()}.log"
            log_path = log_dir / log_name

            cmd = [sys.executable, str(main_py), "--image_folder", str(image_folder)] + extra_list

            print("\n" + "=" * 80)
            print(f"[RUN] dataset={ds} mode={mode} tag={tag}")
            print(f"[LOG] {log_path}")
            print(f"[CMD] {' '.join(cmd)}")
            print("=" * 80 + "\n")

            if args.dry_run:
                continue

            t0 = time.time()
            rc = stream_subprocess(cmd, cwd=repo_root, log_path=log_path, env=env)
            sec = time.time() - t0

            append_summary(summary_csv, {
                "time": datetime.now().isoformat(timespec="seconds"),
                "tag": tag,
                "dataset": ds,
                "mode": mode,
                "image_folder": str(image_folder),
                "returncode": rc,
                "seconds": f"{sec:.3f}",
                "log_file": str(log_path),
                "cmd": " ".join(cmd),
            })

            if rc != 0:
                print(f"[ERROR] main.py returned {rc} (see log: {log_path})", file=sys.stderr)

    print(f"\n[DONE] Summary CSV: {summary_csv}")
    print(f"[DONE] Logs directory: {log_dir}")


if __name__ == "__main__":
    main()

```

### scripts/run_semantic_suite_plus.py

```python
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse, csv, os, re, shlex, sys, time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import subprocess
from typing import Optional, Tuple, List
import numpy as np

try:
    import matplotlib.pyplot as plt
    HAS_PLT = True
except Exception:
    HAS_PLT = False

def ts() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")

def repo_root_from_script() -> Path:
    return Path(__file__).resolve().parents[1]

def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def safe_float(x: str):
    try: return float(x)
    except: return None

@dataclass
class Traj:
    t: np.ndarray
    p: np.ndarray
    q: Optional[np.ndarray] = None  # xyzw

def quat_normalize(q: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(q, axis=-1, keepdims=True) + 1e-12
    return q / n

def quat_conj(q: np.ndarray) -> np.ndarray:
    qc = q.copy()
    qc[..., :3] *= -1.0
    return qc

def quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    ax, ay, az, aw = a[...,0], a[...,1], a[...,2], a[...,3]
    bx, by, bz, bw = b[...,0], b[...,1], b[...,2], b[...,3]
    x = aw*bx + ax*bw + ay*bz - az*by
    y = aw*by - ax*bz + ay*bw + az*bx
    z = aw*bz + ax*by - ay*bx + az*bw
    w = aw*bw - ax*bx - ay*by - az*bz
    return np.stack([x,y,z,w], axis=-1)

def quat_to_rot_angle_deg(q_rel: np.ndarray) -> np.ndarray:
    q_rel = quat_normalize(q_rel)
    w = np.clip(q_rel[...,3], -1.0, 1.0)
    ang = 2.0 * np.arccos(w)
    return np.degrees(ang)

def compute_basic_metrics(traj: Traj, jump_trans_thresh: float, jump_rot_thresh_deg: float) -> dict:
    p = traj.p
    N = p.shape[0]
    if N < 2:
        return {"n_poses": N}

    dp = p[1:] - p[:-1]
    step_trans = np.linalg.norm(dp, axis=1)
    out = {
        "n_poses": int(N),
        "path_length_m": float(step_trans.sum()),
        "loop_trans_err_m": float(np.linalg.norm(p[-1] - p[0])),
        "step_trans_mean_m": float(np.mean(step_trans)),
        "step_trans_p95_m": float(np.percentile(step_trans, 95)),
        "step_trans_max_m": float(np.max(step_trans)),
        "jump_trans_count": int(np.sum(step_trans > jump_trans_thresh)),
        "loop_rot_err_deg": "",
        "step_rot_mean_deg": "",
        "step_rot_p95_deg": "",
        "step_rot_max_deg": "",
        "jump_rot_count": "",
    }

    if traj.q is not None and traj.q.shape[0] == N:
        q = quat_normalize(traj.q)
        q_rel = quat_mul(q[1:], quat_conj(q[:-1]))
        step_rot = quat_to_rot_angle_deg(q_rel)
        q_loop = quat_mul(q[-1], quat_conj(q[0]))
        out["loop_rot_err_deg"] = float(quat_to_rot_angle_deg(q_loop)[()])
        out["step_rot_mean_deg"] = float(np.mean(step_rot))
        out["step_rot_p95_deg"] = float(np.percentile(step_rot, 95))
        out["step_rot_max_deg"] = float(np.max(step_rot))
        out["jump_rot_count"] = int(np.sum(step_rot > jump_rot_thresh_deg))

    return out

TRAJ_PATTERNS = [
    r".*traj.*\.(txt|csv)$",
    r".*trajectory.*\.(txt|csv)$",
    r".*pose.*\.(txt|csv)$",
    r".*poses.*\.(txt|csv)$",
    r".*tum.*\.(txt|csv)$",
    r".*kitti.*\.(txt|csv)$",
    r".*estimated.*\.(txt|csv)$",
    r".*cam.*pose.*\.(txt|csv)$",
]

def list_recent_files(root: Path, start_epoch: float, regex_list: List[str]) -> List[Path]:
    regs = [re.compile(r, re.IGNORECASE) for r in regex_list]
    hits = []
    if not root.exists():
        return hits
    for p in root.rglob("*"):
        if not p.is_file(): continue
        try:
            if p.stat().st_mtime < start_epoch: continue
        except: 
            continue
        if any(r.match(p.name) for r in regs):
            hits.append(p)
    hits.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    return hits

def try_parse_traj_file(path: Path) -> Optional[Traj]:
    try:
        lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    except Exception:
        return None
    rows = []
    for ln in lines:
        s = ln.strip()
        if not s or s.startswith("#"): 
            continue
        parts = re.split(r"[,\s]+", s)
        vals = [safe_float(x) for x in parts]
        if any(v is None for v in vals): 
            continue
        rows.append([float(v) for v in vals])
    if len(rows) < 2:
        return None
    arr = np.array(rows, dtype=np.float64)

    if arr.shape[1] >= 8:  # TUM
        return Traj(t=arr[:,0], p=arr[:,1:4], q=arr[:,4:8])
    if arr.shape[1] == 12: # KITTI
        p = np.stack([arr[:,3], arr[:,7], arr[:,11]], axis=1)
        t = np.arange(p.shape[0], dtype=np.float64)
        return Traj(t=t, p=p, q=None)
    if arr.shape[1] == 4:  # t xyz
        return Traj(t=arr[:,0], p=arr[:,1:4], q=None)
    if arr.shape[1] == 3:  # xyz
        p = arr[:,0:3]
        t = np.arange(p.shape[0], dtype=np.float64)
        return Traj(t=t, p=p, q=None)

    return None

def discover_and_parse_traj(run_dir: Path, start_epoch: float) -> Tuple[str, Optional[Traj]]:
    cand = list_recent_files(run_dir, start_epoch, TRAJ_PATTERNS)
    for f in cand[:50]:
        traj = try_parse_traj_file(f)
        if traj is not None and traj.p.shape[0] >= 2:
            return str(f), traj
    return "", None

def save_plots(traj: Traj, out_dir: Path, prefix: str) -> dict:
    out = {"plot_traj":"", "plot_step_trans":"", "plot_step_rot":""}
    if not HAS_PLT:
        return out
    ensure_dir(out_dir)
    p = traj.p
    x = p[:,0]
    y = p[:,2] if p.shape[1] >= 3 else p[:,1]

    fig = plt.figure()
    plt.plot(x, y)
    plt.axis("equal")
    plt.title("Trajectory (top-down)")
    plt.xlabel("x")
    plt.ylabel("z" if p.shape[1] >= 3 else "y")
    fp = out_dir / f"{prefix}__traj_xy.png"
    fig.savefig(fp, dpi=200, bbox_inches="tight")
    plt.close(fig)
    out["plot_traj"] = str(fp)

    dp = p[1:] - p[:-1]
    step_trans = np.linalg.norm(dp, axis=1)
    fig = plt.figure()
    plt.plot(step_trans)
    plt.title("Step translation (m)")
    plt.xlabel("step idx")
    plt.ylabel("m")
    fp = out_dir / f"{prefix}__step_trans.png"
    fig.savefig(fp, dpi=200, bbox_inches="tight")
    plt.close(fig)
    out["plot_step_trans"] = str(fp)

    if traj.q is not None and traj.q.shape[0] == p.shape[0]:
        q = quat_normalize(traj.q)
        q_rel = quat_mul(q[1:], quat_conj(q[:-1]))
        step_rot = quat_to_rot_angle_deg(q_rel)
        fig = plt.figure()
        plt.plot(step_rot)
        plt.title("Step rotation (deg)")
        plt.xlabel("step idx")
        plt.ylabel("deg")
        fp = out_dir / f"{prefix}__step_rot.png"
        fig.savefig(fp, dpi=200, bbox_inches="tight")
        plt.close(fig)
        out["plot_step_rot"] = str(fp)

    return out

def stream_subprocess(cmd: list[str], cwd: Path, log_path: Path, env: dict[str,str]) -> int:
    ensure_dir(log_path.parent)
    with log_path.open("w", encoding="utf-8") as f:
        f.write(f"[CMD] {' '.join(cmd)}\n")
        f.write(f"[CWD] {cwd}\n")
        f.write(f"[TIME] {datetime.now().isoformat()}\n")
        f.write("-"*80 + "\n")
        f.flush()

        proc = subprocess.Popen(
            cmd, cwd=str(cwd),
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1, universal_newlines=True, env=env
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            sys.stdout.write(line); sys.stdout.flush()
            f.write(line); f.flush()
        return proc.wait()

def append_csv(csv_path: Path, row: dict, fieldnames: List[str]) -> None:
    ensure_dir(csv_path.parent)
    write_header = not csv_path.exists()
    with csv_path.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header: w.writeheader()
        w.writerow({k: row.get(k, "") for k in fieldnames})

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo_root", type=str, default="")
    ap.add_argument("--main_py", type=str, default="main.py")
    ap.add_argument("--data_root", type=str, default="DATA")
    ap.add_argument("--datasets", nargs="+", default=["control_0100-0199", "corridor_0300-0399"])
    ap.add_argument("--modes", nargs="+", default=["baseline", "semantic"])
    ap.add_argument("--baseline_args", type=str, default="")
    ap.add_argument("--semantic_args", type=str, default="")
    ap.add_argument("--tag", type=str, default="no_tag")
    ap.add_argument("--log_dir", type=str, default="LOG")
    ap.add_argument("--run_root", type=str, default="RUNS")
    ap.add_argument("--metrics_csv", type=str, default="LOG/semantic_suite_metrics.csv")
    ap.add_argument("--jump_trans_thresh", type=float, default=0.5)
    ap.add_argument("--jump_rot_thresh_deg", type=float, default=30.0)

    # 关键升级：强制让 main.py 输出落盘（利用 main.py 自带 --log_results/--log_path）
    ap.add_argument("--force_log_results", action="store_true", default=True,
                    help="Force append --log_results --log_path <run_dir> to main.py.")
    ap.add_argument("--no_force_log_results", action="store_true",
                    help="Disable force log_results/log_path injection.")
    ap.add_argument("passthrough", nargs=argparse.REMAINDER)
    args = ap.parse_args()

    repo_root = Path(args.repo_root).resolve() if args.repo_root else repo_root_from_script()
    main_py = (repo_root / args.main_py).resolve()
    if not main_py.exists():
        print(f"[ERROR] main.py not found: {main_py}", file=sys.stderr)
        sys.exit(2)

    data_root = (repo_root / args.data_root).resolve()
    if not data_root.exists():
        print(f"[ERROR] data_root not found: {data_root}", file=sys.stderr)
        sys.exit(2)

    log_dir = (repo_root / args.log_dir).resolve(); ensure_dir(log_dir)
    run_root = (repo_root / args.run_root).resolve(); ensure_dir(run_root)
    metrics_csv = (repo_root / args.metrics_csv).resolve()

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"

    passthrough = args.passthrough
    if passthrough and passthrough[0] == "--":
        passthrough = passthrough[1:]

    mode_to_args = {"baseline": args.baseline_args, "semantic": args.semantic_args}

    fieldnames = [
        "time","tag","dataset","mode","returncode","seconds",
        "image_folder","run_dir","log_file","traj_file",
        "n_poses","path_length_m","loop_trans_err_m","loop_rot_err_deg",
        "step_trans_mean_m","step_trans_p95_m","step_trans_max_m",
        "step_rot_mean_deg","step_rot_p95_deg","step_rot_max_deg",
        "jump_trans_count","jump_rot_count",
        "plot_traj","plot_step_trans","plot_step_rot",
        "cmd",
    ]

    for ds in args.datasets:
        ds_dir = data_root / ds
        if not ds_dir.exists():
            print(f"[WARN] dataset not found: {ds_dir}")
            continue

        for mode in args.modes:
            extra = mode_to_args.get(mode, "")
            extra_list = shlex.split(extra)

            run_stamp = ts()
            run_dir = run_root / args.tag / ds / mode / run_stamp
            ensure_dir(run_dir)

            log_path = log_dir / f"{args.tag}__{ds}__{mode}__{run_stamp}.log"

            cmd = [sys.executable, str(main_py), "--image_folder", str(ds_dir)]
            cmd += extra_list
            cmd += passthrough

            # 注入 log_results/log_path，让 main.py 把轨迹/结果稳定输出到 run_dir
            force = args.force_log_results and (not args.no_force_log_results)
            if force:
                if "--log_results" not in cmd:
                    cmd += ["--log_results"]
                if "--log_path" not in cmd:
                    cmd += ["--log_path", str(run_dir / "poses.txt")]

            print("\n" + "="*80)
            print(f"[RUN] tag={args.tag} dataset={ds} mode={mode}")
            print(f"[RUN_DIR] {run_dir}")
            print(f"[LOG] {log_path}")
            print(f"[CMD] {' '.join(cmd)}")
            print("="*80 + "\n")

            start_epoch = time.time()
            t0 = time.time()
            rc = stream_subprocess(cmd, cwd=repo_root, log_path=log_path, env=env)
            sec = time.time() - t0

            traj_file, traj = discover_and_parse_traj(run_dir, start_epoch)

            metrics = {
                "n_poses":"","path_length_m":"","loop_trans_err_m":"","loop_rot_err_deg":"",
                "step_trans_mean_m":"","step_trans_p95_m":"","step_trans_max_m":"",
                "step_rot_mean_deg":"","step_rot_p95_deg":"","step_rot_max_deg":"",
                "jump_trans_count":"","jump_rot_count":"",
            }
            plots = {"plot_traj":"","plot_step_trans":"","plot_step_rot":""}

            if traj is not None:
                metrics = compute_basic_metrics(traj, args.jump_trans_thresh, args.jump_rot_thresh_deg)
                plots = save_plots(traj, run_dir, f"{ds}__{mode}")

            row = {
                "time": now_iso(),
                "tag": args.tag,
                "dataset": ds,
                "mode": mode,
                "returncode": rc,
                "seconds": f"{sec:.3f}",
                "image_folder": str(ds_dir),
                "run_dir": str(run_dir),
                "log_file": str(log_path),
                "traj_file": traj_file,
                "cmd": " ".join(cmd),
            }
            row.update(metrics)
            row.update(plots)
            append_csv(metrics_csv, row, fieldnames)

            if traj is None:
                print(f"[WARN] 仍未解析到轨迹文件。请到 run_dir 看 main.py 实际输出了什么文件：{run_dir}")

    print(f"\n[DONE] {metrics_csv}")

if __name__ == "__main__":
    main()

```

### scripts/undistort.py

```python
import cv2
import numpy as np
import os
from glob import glob

K = np.array([
    [458.654, 0, 367.215],
    [0, 457.296, 248.375],
    [0,  0,  1]
], dtype=np.float32)

# Radial-tangential distortion coefficients
dist_coeffs = np.array([-0.28340811, 0.07395907, 0.00019359, 1.76187114e-05], dtype=np.float32)

datasets = ["MH_01_easy", "MH_02_easy", "MH_03_medium", "MH_04_difficult", "MH_05_difficult", 
           "V1_01_easy", "V1_02_medium", "V1_03_difficult", "V2_01_easy", "V2_02_medium", 
           "V2_03_difficult", "V2_04_difficult", "V2_05_difficult"]

# === Input/output folders ===
for dataset in datasets:
    input_folder = "/home/<user>/Documents/MASt3R-SLAM/datasets/euroc/" + dataset + "/mav0/cam0/data"
    output_folder = "/home/<user>/Documents/MASt3R-SLAM/datasets/euroc/" + dataset + "/mav0/cam0/data_rectified"

    # Create output directory if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Get all image paths (adjust extension if needed)
    image_paths = glob(os.path.join(input_folder, '*.png'))

    for img_path in image_paths:
        img = cv2.imread(img_path)
        if img is None:
            print(f"Warning: Failed to read {img_path}")
            continue

        # Undistort
        undistorted = cv2.undistort(img, K, dist_coeffs)

        # Save with same filename in output folder
        filename = os.path.basename(img_path)
        out_path = os.path.join(output_folder, filename)
        cv2.imwrite(out_path, undistorted)

        print(f"Saved: {out_path}")

```

### tools/sem_pretrain/extract_dino_feats.py

```python
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from PIL import Image

try:
    from tqdm import tqdm
except Exception:
    tqdm = None

import torchvision.transforms as T


# ======= 你的离线路径（按你提供的）=======
DINO_REPO_DIR = "/media/omnisky/18/cst1/cache/torch/hub/facebookresearch_dinov2_main"
DINO_CKPT = "/media/omnisky/18/cst1/cache/torch/hub/checkpoints/dinov2_vitb14_pretrain.pth"
DINO_MODEL_NAME = "dinov2_vitb14"


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def list_images(image_dir: str, recursive: bool = True) -> List[str]:
    root = Path(image_dir)
    if not root.exists():
        raise FileNotFoundError(f"image_dir not found: {image_dir}")

    if root.is_file():
        # allow single image
        if root.suffix.lower() in IMG_EXTS:
            return [str(root)]
        raise ValueError(f"Provided image_dir is a file but not an image: {image_dir}")

    pattern = "**/*" if recursive else "*"
    files = []
    for p in root.glob(pattern):
        if p.is_file() and p.suffix.lower() in IMG_EXTS:
            files.append(str(p))

    files.sort()
    if len(files) == 0:
        raise RuntimeError(f"No images found under: {image_dir}")
    return files


def build_transform(image_size: int) -> T.Compose:
    # DINOv2 常用 ImageNet normalize
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    return T.Compose([
        T.Resize(image_size, interpolation=T.InterpolationMode.BICUBIC),
        T.CenterCrop(image_size),
        T.ToTensor(),
        T.Normalize(mean=mean, std=std),
    ])


def load_dinov2(device: str = "cuda") -> torch.nn.Module:
    """
    关键点：
    - torch.hub.load(..., source="local") -> 不联网、不访问 GitHub
    - pretrained=False -> 不触发在线权重下载
    - 手动 load_state_dict 读取本地 ckpt
    """
    if not os.path.isdir(DINO_REPO_DIR):
        raise FileNotFoundError(
            f"DINO repo dir not found: {DINO_REPO_DIR}\n"
            f"Expected hubconf.py under it."
        )
    if not os.path.isfile(os.path.join(DINO_REPO_DIR, "hubconf.py")):
        raise FileNotFoundError(
            f"hubconf.py not found under: {DINO_REPO_DIR}\n"
            f"Make sure this is the dinov2 repo root."
        )
    if not os.path.isfile(DINO_CKPT):
        raise FileNotFoundError(f"DINO checkpoint not found: {DINO_CKPT}")

    model = torch.hub.load(
        DINO_REPO_DIR,
        DINO_MODEL_NAME,
        source="local",
        pretrained=False,
    )

    state = torch.load(DINO_CKPT, map_location="cpu")
    # 有些权重是 {"model": state_dict, ...}
    if isinstance(state, dict) and "model" in state and isinstance(state["model"], dict):
        state = state["model"]

    missing, unexpected = model.load_state_dict(state, strict=False)
    if len(unexpected) > 0:
        print(f"[WARN] unexpected keys: {unexpected[:10]}{'...' if len(unexpected) > 10 else ''}")
    if len(missing) > 0:
        print(f"[WARN] missing keys: {missing[:10]}{'...' if len(missing) > 10 else ''}")

    model = model.to(device).eval()
    return model


@torch.no_grad()
def extract_features(
    model: torch.nn.Module,
    image_paths: List[str],
    device: str,
    image_size: int,
    batch_size: int,
) -> Tuple[np.ndarray, List[str]]:
    tfm = build_transform(image_size)

    feats = []
    ok_paths = []

    iterator = range(0, len(image_paths), batch_size)
    if tqdm is not None:
        iterator = tqdm(iterator, total=(len(image_paths) + batch_size - 1) // batch_size, desc="Extract")

    for i in iterator:
        batch_paths = image_paths[i:i + batch_size]
        imgs = []
        valid_paths = []

        for p in batch_paths:
            try:
                img = Image.open(p).convert("RGB")
                imgs.append(tfm(img))
                valid_paths.append(p)
            except Exception as e:
                print(f"[WARN] skip unreadable image: {p} ({e})")

        if len(imgs) == 0:
            continue

        x = torch.stack(imgs, dim=0).to(device, non_blocking=True)

        # DINOv2 推荐接口：forward_features -> dict，取 x_norm_clstoken
        if hasattr(model, "forward_features"):
            out = model.forward_features(x)
            if isinstance(out, dict) and "x_norm_clstoken" in out:
                f = out["x_norm_clstoken"]
            elif isinstance(out, dict) and "x_prenorm" in out:
                # 兜底：不保证存在
                f = out["x_prenorm"][:, 0]
            else:
                # 再兜底
                f = model(x)
        else:
            f = model(x)

        # 统一到 [B, D]
        if f.dim() == 3:
            # [B, T, D] -> cls token
            f = f[:, 0, :]
        f = f.detach().float().cpu().numpy()

        feats.append(f)
        ok_paths.extend(valid_paths)

    feats = np.concatenate(feats, axis=0) if len(feats) > 0 else np.zeros((0, 0), dtype=np.float32)
    return feats, ok_paths


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dir", type=str, required=True, help="image folder (recursive) or a single image")
    parser.add_argument("--out", type=str, required=True, help="output .npy path")
    parser.add_argument("--device", type=str, default="cuda", help="cuda / cpu / cuda:0 ...")
    parser.add_argument("--image_size", type=int, default=518, help="resize+crop size (DINOv2 commonly 518)")
    parser.add_argument("--batch_size", type=int, default=32, help="batch size")
    parser.add_argument("--no_recursive", action="store_true", help="disable recursive scan")
    parser.add_argument("--save_paths", action="store_true", help="also save a .txt with image paths next to .npy")
    args = parser.parse_args()

    device = args.device
    if device.startswith("cuda") and (not torch.cuda.is_available()):
        print("[WARN] CUDA not available, falling back to cpu")
        device = "cpu"

    image_paths = list_images(args.image_dir, recursive=(not args.no_recursive))
    print(f"[INFO] found {len(image_paths)} images")

    model = load_dinov2(device=device)
    feats, ok_paths = extract_features(
        model=model,
        image_paths=image_paths,
        device=device,
        image_size=args.image_size,
        batch_size=args.batch_size,
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(out_path), feats.astype(np.float32))
    print(f"[INFO] saved feats: {feats.shape} -> {out_path}")

    if args.save_paths:
        txt_path = out_path.with_suffix(".paths.txt")
        with open(txt_path, "w", encoding="utf-8") as f:
            for p in ok_paths:
                f.write(p + "\n")
        print(f"[INFO] saved paths -> {txt_path}")


if __name__ == "__main__":
    main()

```

### tools/sem_pretrain/train_metric.py

```python
import os
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


class PairDataset(Dataset):
    def __init__(self, feats, pos_win=2, neg_min=50, hard_k=20):
        self.feats = torch.from_numpy(feats).float()
        self.feats = nn.functional.normalize(self.feats, dim=-1)
        self.N = self.feats.shape[0]
        self.pos_win = int(pos_win)
        self.neg_min = int(neg_min)

        # hard negatives: top-k similar but far in time
        sim = (self.feats @ self.feats.t()).cpu().numpy()
        self.hard_negs = []
        for i in range(self.N):
            cand = np.argsort(-sim[i])
            hard = []
            for j in cand:
                if abs(i - j) >= self.neg_min:
                    hard.append(int(j))
                if len(hard) >= hard_k:
                    break
            if len(hard) == 0:
                hard = [min(self.N - 1, i + self.neg_min)]
            self.hard_negs.append(hard)

    def __len__(self):
        return self.N

    def __getitem__(self, i):
        # positive: nearby frame
        lo = max(0, i - self.pos_win)
        hi = min(self.N - 1, i + self.pos_win)
        j = np.random.randint(lo, hi + 1)
        if j == i:
            j = min(self.N - 1, i + 1)

        # negative: mostly hard negative
        if np.random.rand() < 0.8:
            k = int(np.random.choice(self.hard_negs[i]))
        else:
            while True:
                k = np.random.randint(0, self.N)
                if abs(i - k) >= self.neg_min:
                    break

        return self.feats[i], self.feats[j], self.feats[k]


class MetricHead(nn.Module):
    def __init__(self, dim, out_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(inplace=True),
            nn.Linear(dim, out_dim),
        )

    def forward(self, x):
        x = self.net(x)
        return nn.functional.normalize(x, dim=-1)


def info_nce(a, p, n, tau=0.07):
    pos = (a * p).sum(-1, keepdim=True) / tau
    neg = (a * n).sum(-1, keepdim=True) / tau
    logits = torch.cat([pos, neg], dim=1)  # [B,2]
    labels = torch.zeros(a.size(0), dtype=torch.long, device=a.device)
    return nn.CrossEntropyLoss()(logits, labels)


def main(feats_npy, out_ckpt, device="cuda", epochs=10, bs=64, lr=1e-3, num_workers=2):
    data = np.load(feats_npy, allow_pickle=True).item()
    feats = data["feats"]
    dim = feats.shape[1]
    N = feats.shape[0]

    neg_min = max(20, N // 5)
    ds = PairDataset(feats, pos_win=2, neg_min=neg_min, hard_k=min(50, max(5, N // 2)))

    # 关键：别让 batch_size > N 且 drop_last=True 导致 0 step
    bs = int(min(bs, N))
    dl = DataLoader(
        ds,
        batch_size=bs,
        shuffle=True,
        num_workers=int(num_workers),
        drop_last=False,
        pin_memory=True,
    )

    head = MetricHead(dim, out_dim=256).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=float(lr))

    head.train()
    for ep in range(int(epochs)):
        losses = []
        for a, p, n in tqdm(dl, desc=f"ep{ep}"):
            a, p, n = a.to(device), p.to(device), n.to(device)
            za, zp, zn = head(a), head(p), head(n)
            loss = info_nce(za, zp, zn)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            losses.append(float(loss.item()))

        if len(losses) == 0:
            raise RuntimeError(
                f"No training steps executed. N={N}, bs={bs}. "
                f"Check DataLoader settings."
            )

        print(f"epoch {ep} loss={np.mean(losses):.4f} (steps={len(losses)})")

    os.makedirs(os.path.dirname(out_ckpt), exist_ok=True)
    torch.save({"state_dict": head.state_dict(), "dim": dim}, out_ckpt)
    print("saved:", out_ckpt)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--feats", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--num_workers", type=int, default=2)
    args = ap.parse_args()
    main(args.feats, args.out, args.device, args.epochs, args.bs, args.lr, args.num_workers)

```

### tools/sem_pretrain/vis_retrieval.py

```python

```

### vggt_slam/__init__.py

```python

```

### vggt_slam/frame_overlap.py

```python
import argparse
import torch
import numpy as np
import cv2

# from core.raft import RAFT
# from core.utils import flow_viz
# from core.utils.utils import InputPadder

device = 'cuda'

def load_raft():
    print("Loading RAFT ...")
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default = "/home/<user>/RAFT/models/raft-sintel.pth")
    parser.add_argument('--path', default = "not_used")
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--alternate_corr', action='store_true', help='use efficent correlation implementation')
    args, _ = parser.parse_known_args()
    model = torch.nn.DataParallel(RAFT(args))
    model.load_state_dict(torch.load(args.model))

    model = model.module
    model.to(device)
    model.eval()
    print("RAFT Loaded")
    return model

def get_raft_image(image):
    img = image.astype(np.uint8)
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img[None].to(device)

def get_uniform_points(h,w, delta):
    uniform_points = []
    for r in range(h):
        if r%delta == 0:
            for c in range(w):
                if c%delta == 0:
                    uniform_points.append([c,r])
    return uniform_points

def get_sparse_flow(img, flo, origin_points):
    flo = np.transpose(flo, (1,2,0))
    h,w,_ = img.shape
    flow_coords = []
    p0 = []
    p1 = []
    for point in origin_points:
        if point[0] < 0 or point[1] < 0 or point[0] >= w or point[1] >= h:
            point[0] = 0
            point[1] = 0

        p0.append([point[0], point[1]])
        p1.append([point[0]+(flo[point[1],point[0],0]), point[1]+(flo[point[1],point[0],1])])
        flow_coords.append(([point[1], point[0]], [point[1]+(flo[point[1],point[0],1]), point[0]+(flo[point[1],point[0],0])]))

    p0 = np.expand_dims(np.asarray(p0, dtype=np.float64), axis=0)
    p1 = np.expand_dims(np.asarray(p1, dtype=np.float64), axis=0)

    return p0, p1

def visualize_flow(img, flow_up, origin_points):
    p0, p1 = get_sparse_flow(img, flow_up, origin_points)
    img = np.copy(img)
    for i in range(p0.shape[1]):
        p0_x = p0[0,i,0]
        p0_y = p0[0,i,1]
        p1_x = p1[0,i,0]
        p1_y = p1[0,i,1]
        color = (0,0,0)
        cv2.arrowedLine(img, (int(p0_x), int(p0_y)), (int(p1_x), int(p1_y)), color=color, thickness=2, tipLength=1)
    cv2.imshow('image', img[:, :, [2,1,0]]/255.0)
    cv2.waitKey()

class FrameTrackerRaft:
    def __init__(self):
        self.raft_model = load_raft()
        self.count = 0
        self.last_kf = None
    
    def get_optical_flow(self, image):
        """
        get optical flow of points from img1 to img2
        """
        image1 = get_raft_image(self.last_kf)
        image2 = get_raft_image(image)

        with torch.no_grad():
            padder = InputPadder(image1.shape)
            image1, image2 = padder.pad(image1, image2)
            _, flow_up = self.raft_model(image1, image2, iters=20, test_mode=True)
        return flow_up[0,:,:].cpu().numpy()

    def compute_disparity(self, image, min_disparity, visualize=False):
        if self.last_kf is None:
            self.last_kf = image
            # First frame, no previous frame to compare to, return True so the first frame is a keyframe.
            return True
        flow = self.get_optical_flow(image)
        flow_magnitude = np.linalg.norm(flow, axis=0)

        mean_disparity = np.mean(flow_magnitude)

        if visualize:
            visualize_flow(image, flow, get_uniform_points(image.shape[0], image.shape[1], 15))

        if mean_disparity > min_disparity:
            self.last_kf = image
            # cv2.imshow("img",image)
            # cv2.waitKey()
            self.count += 1
            return True
        return False

class FrameTracker:
    def __init__(self):
        self.last_kf = None
        self.kf_pts = None
        self.kf_gray = None

    def initialize_keyframe(self, image):
        self.last_kf = image
        self.kf_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        self.kf_pts = cv2.goodFeaturesToTrack(
            self.kf_gray,
            maxCorners=1000,
            qualityLevel=0.01,
            minDistance=8,
            blockSize=7
        )

    def compute_disparity(self, image, min_disparity, visualize=False):
        if self.last_kf is None or self.kf_pts is None or len(self.kf_pts) < 10:
            self.initialize_keyframe(image)
            return True

        curr_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Track keyframe points into current frame
        next_pts, status, _ = cv2.calcOpticalFlowPyrLK(
            self.kf_gray, curr_gray, self.kf_pts, None,
            winSize=(21, 21), maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
        )

        status = status.flatten()
        good_kf = self.kf_pts[status == 1]
        good_next = next_pts[status == 1]

        if len(good_kf) < 10:
            self.initialize_keyframe(image)
            return True

        # Measure displacement from keyframe to current frame
        displacement = np.linalg.norm(good_next - good_kf, axis=1)
        mean_disparity = np.mean(displacement)

        if visualize:
            vis = image.copy()
            for p1, p2 in zip(good_kf, good_next):
                p1 = tuple(p1.ravel().astype(int))
                p2 = tuple(p2.ravel().astype(int))
                cv2.arrowedLine(vis, p1, p2, color=(0, 255, 0), thickness=1, tipLength=0.3)
            cv2.imshow("Optical Flow", vis)
            cv2.waitKey(1)

        if mean_disparity > min_disparity:
            self.initialize_keyframe(image)
            return True
        else:
            return False
```

### vggt_slam/gradio_viewer.py

```python
import trimesh
import numpy as np
from pathlib import Path


class TrimeshViewer:
    def __init__(self):
        self.scene = trimesh.Scene()
        self.frame_size = 0.1  # Scale of camera frame axes

    def add_camera_pose(self, cam2world: np.ndarray):
        """Add camera coordinate frame (as axis) to scene."""
        T = cam2world

        # Trimesh's built-in coordinate frame
        cam_frame = trimesh.creation.axis(origin_size=0.01, axis_length=self.frame_size)
        cam_frame.apply_transform(T)

        self.scene.add_geometry(cam_frame)

    def add_point_cloud(self, points: np.ndarray, colors: np.ndarray = None):
        """
        points: (N, 3)
        colors: (N, 3) uint8 or float
        """
        point_cloud = trimesh.points.PointCloud(points, colors=colors)
        self.scene.add_geometry(point_cloud)

    def export(self, out_path="output.glb") -> str:
        """Export the scene as a GLB file (for Hugging Face display)."""
        out_path = Path(out_path)
        self.scene.export(out_path)
        return str(out_path)

```

### vggt_slam/graph.py

```python
# vggt_slam/graph.py
import numpy as np
import gtsam
from gtsam import noiseModel
from gtsam.symbol_shorthand import X


class PoseGraph:
    """
    SL(4) pose graph (15 DoF).

    保护策略：
    1) 输入矩阵构造 SL4 前：过滤 NaN/Inf、奇异、离谱尺度、离谱 cond。
    2) det<0：不能用 H=-H（4x4 det 不变），必须乘 det=-1 的反射矩阵（左右乘都尝试）。
    3) gtsam.SL4(H) 构造：try/except，失败直接 SKIP，不让进程崩。
    4) optimize：若优化内部因 SL4 归一化/退化报错，fallback 为“增量加因子，谁炸跳过谁”，最终仍给出 result。
    """

    def __init__(self):
        self.graph = gtsam.NonlinearFactorGraph()
        self.initial = gtsam.Values()
        self.result = None

        # Base noise for SL4 (15 DoF)
        self._relative_sigmas = 0.05 * np.ones(15, dtype=float)
        self.relative_noise = noiseModel.Diagonal.Sigmas(self._relative_sigmas)
        self.anchor_noise = noiseModel.Diagonal.Sigmas([1e-6] * 15)

        self.num_loop_closures = 0

        # ---- SL4 safety knobs ----
        self.det_eps = 1e-10              # near singular threshold
        self.fix_negative_det = True      # det<0 -> reflect to make det>0
        self.skip_near_singular = True    # skip if |det| too small

        # det magnitude sanity (before det=1 normalize)
        self.det_abs_min = 1e-6
        self.det_abs_max = 1e6

        # condition number sanity
        self.cond_max = 1e8

        # matrix magnitude sanity (Frobenius norm)
        self.frob_max = 1e4

        # ---- Robust noise ----
        self.use_robust_between = True
        self.huber_k = 1.345

        # ---- Conservative LM knobs ----
        self.lm_max_iters = 50
        self.lm_lambda_initial = 1e3
        self.lm_lambda_factor = 10.0
        self.lm_diagonal_damping = True
        self.lm_verbosity = "ERROR"
        self.verbosity = "ERROR"

        # optimize fallback
        self.enable_safe_incremental_opt = True
        self.safe_inc_max_iters = 30   # 每次增量优化的最大迭代（别太大，避免慢）
        self.safe_inc_lambda_initial = 1e6  # 更保守，避免 retraction 走飞

    # ---------------- utils ----------------

    @staticmethod
    def _finite(x) -> bool:
        return np.isfinite(np.asarray(x)).all()

    def _robustify(self, base_noise):
        if not self.use_robust_between:
            return base_noise
        try:
            huber = noiseModel.mEstimator.Huber.Create(self.huber_k)
            return noiseModel.Robust.Create(huber, base_noise)
        except Exception:
            return base_noise

    def _normalize_det_to_one(self, H: np.ndarray, det: float) -> np.ndarray:
        """
        det(sH) = s^4 det(H)
        令 |det| -> 1：s = |det(H)|^(-1/4)
        注意：只做尺度归一化，不改变 det 符号。
        """
        scale = float(abs(det) ** 0.25)
        if scale <= 0 or (not np.isfinite(scale)):
            return H
        return H / scale

    def _reflect_to_positive_det(self, H: np.ndarray, det: float, ctx: str):
        """
        det<0 时，用 det=-1 的反射矩阵 R 翻正：
          det(RH) = det(R)*det(H) = (-1)*det(H) > 0

        为了更稳：同时尝试左乘 R@H 和右乘 H@R（各 3 个轴反射），挑 cond 更小的。
        """
        if det >= 0:
            return H, det

        if not self.fix_negative_det:
            print(f"[PoseGraph] {ctx}: det<0 det={det:.6g} (fix_negative_det=False) -> SKIP")
            return None, None

        Rs = [
            np.diag([-1.0, 1.0, 1.0, 1.0]),
            np.diag([1.0, -1.0, 1.0, 1.0]),
            np.diag([1.0, 1.0, -1.0, 1.0]),
        ]

        best_H = None
        best_det = None
        best_cond = np.inf
        best_tag = ""

        for i, R in enumerate(Rs):
            for tag, Hc in ((f"L{i}", R @ H), (f"R{i}", H @ R)):
                detc = float(np.linalg.det(Hc))
                if (not np.isfinite(detc)) or detc <= 0:
                    continue
                try:
                    condc = float(np.linalg.cond(Hc))
                except Exception:
                    condc = np.inf
                if condc < best_cond:
                    best_cond = condc
                    best_H = Hc
                    best_det = detc
                    best_tag = tag

        if best_H is None:
            print(f"[PoseGraph] {ctx}: det<0 det={det:.6g} -> reflect failed -> SKIP")
            return None, None

        print(f"[PoseGraph] {ctx}: det<0 det={det:.6g} -> REFLECT({best_tag}) det={best_det:.6g} (cond={best_cond:.3g})")
        return best_H, best_det

    def _make_sl4(self, H: np.ndarray, ctx: str):
        """
        唯一允许调用 gtsam.SL4(H) 的入口。
        返回 gtsam.SL4 或 None（SKIP）。
        """
        H = np.asarray(H, dtype=float)

        if H.shape != (4, 4):
            raise ValueError(f"[PoseGraph] {ctx}: expected 4x4, got {H.shape}")
        if not self._finite(H):
            print(f"[PoseGraph] {ctx}: matrix has NaN/Inf -> SKIP")
            return None

        frob = float(np.linalg.norm(H, ord="fro"))
        if (not np.isfinite(frob)) or frob > self.frob_max:
            print(f"[PoseGraph] {ctx}: frob too large ({frob:.6g}) -> SKIP")
            return None

        det = float(np.linalg.det(H))
        if not np.isfinite(det):
            print(f"[PoseGraph] {ctx}: det is NaN/Inf -> SKIP")
            return None

        if abs(det) < self.det_eps:
            msg = f"[PoseGraph] {ctx}: det too small (degenerate), det={det:.6g}"
            if self.skip_near_singular:
                print(msg + " -> SKIP")
                return None
            raise RuntimeError(msg)

        # det<0 -> 反射翻正
        if det < 0:
            H, det = self._reflect_to_positive_det(H, det, ctx)
            if H is None:
                return None

        # det magnitude sanity
        if abs(det) < self.det_abs_min or abs(det) > self.det_abs_max:
            print(f"[PoseGraph] {ctx}: |det| out of range (det={det:.6g}) -> SKIP")
            return None

        # condition number sanity
        try:
            cond = float(np.linalg.cond(H))
            if (not np.isfinite(cond)) or cond > self.cond_max:
                print(f"[PoseGraph] {ctx}: cond too large ({cond:.6g}) -> SKIP")
                return None
        except Exception:
            print(f"[PoseGraph] {ctx}: cond() failed -> SKIP")
            return None

        # 归一化 |det| -> 1（尺度稳定）
        H = self._normalize_det_to_one(H, det)
        det2 = float(np.linalg.det(H))
        if (not np.isfinite(det2)) or det2 <= 0:
            print(f"[PoseGraph] {ctx}: det after normalize invalid (det={det2:.6g}) -> SKIP")
            return None

        # 永不让 gtsam.SL4 把进程炸掉
        try:
            return gtsam.SL4(H)
        except Exception as e:
            detx = float(np.linalg.det(H))
            print(f"[PoseGraph] {ctx}: gtsam.SL4 failed: {e} (det={detx:.6g}) -> SKIP")
            return None

    def _scaled_relative_noise(self, weight: float):
        w = float(weight)
        if (not np.isfinite(w)) or w <= 0:
            w = 1.0
        sig = self._relative_sigmas / np.sqrt(w)
        sig = np.maximum(sig, 1e-6)
        return noiseModel.Diagonal.Sigmas(sig)

    def _make_lm_params(self, for_safe_incremental: bool = False) -> gtsam.LevenbergMarquardtParams:
        params = gtsam.LevenbergMarquardtParams()
        if hasattr(params, "setMaxIterations"):
            params.setMaxIterations(int(self.safe_inc_max_iters if for_safe_incremental else self.lm_max_iters))
        if hasattr(params, "setLambdaInitial"):
            params.setLambdaInitial(float(self.safe_inc_lambda_initial if for_safe_incremental else self.lm_lambda_initial))
        if hasattr(params, "setLambdaFactor"):
            params.setLambdaFactor(float(self.lm_lambda_factor))
        if hasattr(params, "setDiagonalDamping"):
            params.setDiagonalDamping(bool(self.lm_diagonal_damping))
        if hasattr(params, "setVerbosityLM"):
            try:
                params.setVerbosityLM(self.lm_verbosity)
            except Exception:
                pass
        if hasattr(params, "setVerbosity"):
            try:
                params.setVerbosity(self.verbosity)
            except Exception:
                pass
        return params

    # ---------------- vertices ----------------

    def add_homography(self, key, global_h):
        ctx = f"add_homography key={key}"
        sl4 = self._make_sl4(global_h, ctx)
        if sl4 is None:
            return
        k = X(int(key))
        if self.initial.exists(k):
            self.initial.update(k, sl4)
        else:
            self.initial.insert(k, sl4)

    # ---------------- edges ----------------

    def add_between_factor(self, key1, key2, relative_h, noise):
        ctx = f"add_between_factor ({key1}->{key2})"
        sl4 = self._make_sl4(relative_h, ctx)
        if sl4 is None:
            return
        k1 = X(int(key1))
        k2 = X(int(key2))
        noise = self._robustify(noise)
        self.graph.add(gtsam.BetweenFactorSL4(k1, k2, sl4, noise))

    def add_between_factor_weighted(self, key1, key2, relative_h, weight: float):
        noise = self._scaled_relative_noise(weight)
        self.add_between_factor(key1, key2, relative_h, noise)

    def add_prior_factor(self, key, global_h, noise=None):
        ctx = f"add_prior_factor key={key}"
        sl4 = self._make_sl4(global_h, ctx)
        if sl4 is None:
            return
        k = X(int(key))
        if noise is None:
            noise = self.anchor_noise
        self.graph.add(gtsam.PriorFactorSL4(k, sl4, noise))

    # ---------------- optimize ----------------

    def _optimize_safe_incremental(self):
        """
        关键 fallback：
        - 逐个尝试把 factor 加入 kept 集合
        - 每加入一个，就用当前 values 优化一次
        - 若某个 factor 导致 optimize 内部崩（比如 SL4 det 变负），则跳过该 factor，继续
        """
        n = self.graph.size()
        if n == 0:
            raise RuntimeError("[PoseGraph] empty factor graph (no edges)")
        if self.initial.size() == 0:
            raise RuntimeError("[PoseGraph] empty initial values")

        params = self._make_lm_params(for_safe_incremental=True)

        kept = []
        values = self.initial
        skipped = []

        for i in range(n):
            fac = self.graph.at(i)

            # 重新构建图（避免需要 pop_back）
            trial = gtsam.NonlinearFactorGraph()
            for f in kept:
                trial.push_back(f)
            trial.push_back(fac)

            try:
                opt = gtsam.LevenbergMarquardtOptimizer(trial, values, params)
                values = opt.optimize()
                kept.append(fac)
            except Exception as e:
                skipped.append(i)
                # 可选：打印更详细
                print(f"[PoseGraph][safe_inc] skip factor idx={i}/{n-1} due to optimize error: {e}")
                continue

        print(f"[PoseGraph][safe_inc] kept={len(kept)}/{n}, skipped={len(skipped)}")
        self.result = values
        return self.result

    def optimize(self):
        """
        正常 optimize 失败时，不再 raise（否则你的 ablation 直接 returncode=1）。
        改为 fallback：安全增量优化，跳过会触发 SL4 崩溃的因子。
        """
        if self.graph.size() == 0:
            raise RuntimeError("[PoseGraph] empty factor graph (no edges)")
        if self.initial.size() == 0:
            raise RuntimeError("[PoseGraph] empty initial values")

        params = self._make_lm_params(for_safe_incremental=False)
        optimizer = gtsam.LevenbergMarquardtOptimizer(self.graph, self.initial, params)

        try:
            self.result = optimizer.optimize()
            return self.result
        except Exception as e:
            print("[PoseGraph] Optimization failed, fallback to safe incremental optimize.")
            print(f"  #values : {self.initial.size()}")
            print(f"  #factors: {self.graph.size()}")
            print(f"  error   : {e}")

            if self.enable_safe_incremental_opt:
                return self._optimize_safe_incremental()

            # 最保底：直接不优化，返回 initial（至少不让流程炸）
            self.result = self.initial
            return self.result

    # ---------------- accessors ----------------

    def get_homography(self, node_id):
        k = X(int(node_id))
        if self.result is not None and self.result.exists(k):
            return self.result.atSL4(k).matrix()
        if self.initial.exists(k):
            return self.initial.atSL4(k).matrix()
        return None

    def increment_loop_closure(self):
        self.num_loop_closures += 1

    def get_num_loops(self):
        return self.num_loop_closures

```

### vggt_slam/graph_se3.py

```python
import numpy as np
import gtsam
from gtsam import noiseModel
from gtsam.symbol_shorthand import X


class PoseGraph:
    """
    SE3 pose graph (Pose3).
    """

    def __init__(self):
        self.graph = gtsam.NonlinearFactorGraph()
        self.initial = gtsam.Values()
        self.result = None

        self._relative_sigmas = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1], dtype=float)
        self.relative_noise = noiseModel.Diagonal.Sigmas(self._relative_sigmas)

        self.anchor_noise = noiseModel.Diagonal.Sigmas([1e-6] * 6)

        self.num_loop_closures = 0

    def add_homography(self, key, pose):
        k = X(int(key))
        P = gtsam.Pose3(pose)
        if self.initial.exists(k):
            self.initial.update(k, P)
        else:
            self.initial.insert(k, P)

    def _scaled_relative_noise(self, weight: float):
        w = float(weight)
        if not np.isfinite(w) or w <= 0:
            w = 1.0
        sig = self._relative_sigmas / np.sqrt(w)
        sig = np.maximum(sig, 1e-12)
        return noiseModel.Diagonal.Sigmas(sig)

    def add_between_factor(self, key1, key2, relative_pose, noise):
        k1 = X(int(key1))
        k2 = X(int(key2))
        P = gtsam.Pose3(relative_pose)
        self.graph.add(gtsam.BetweenFactorPose3(k1, k2, P, noise))

    def add_between_factor_weighted(self, key1, key2, relative_pose, weight: float):
        noise = self._scaled_relative_noise(weight)
        self.add_between_factor(key1, key2, relative_pose, noise)

    def add_prior_factor(self, key, pose, noise):
        k = X(int(key))
        P = gtsam.Pose3(pose)
        self.graph.add(gtsam.PriorFactorPose3(k, P, noise))

    def get_homography(self, node_id):
        k = X(int(node_id))
        if self.result is not None and self.result.exists(k):
            return self.result.atPose3(k).matrix()
        if self.initial.exists(k):
            return self.initial.atPose3(k).matrix()
        return None

    def optimize(self):
        params = gtsam.LevenbergMarquardtParams()
        params.setVerbosityLM("ERROR")
        params.setVerbosity("ERROR")
        optimizer = gtsam.LevenbergMarquardtOptimizer(self.graph, self.initial, params)
        self.result = optimizer.optimize()
        return self.result

    def print_estimates(self):
        if self.result is None:
            print("[PoseGraph-SE3] no result yet.")
            return
        keys = list(self.result.keys())
        keys.sort()
        for k in keys:
            try:
                T = self.result.atPose3(k).matrix()
                print(k, "\n", T)
            except Exception:
                pass

    def increment_loop_closure(self):
        self.num_loop_closures += 1

    def get_num_loops(self):
        return self.num_loop_closures

```

### vggt_slam/h_solve.py

```python
import open3d as o3d
import numpy as np
import torch
from scipy.linalg import null_space

def to_homogeneous(X):
    return np.hstack([X, np.ones((X.shape[0], 1))])

def apply_homography(H, X, debug=False):
    X_h = to_homogeneous(X)
    X_trans = (H @ X_h.T).T
    if debug:
        print(X_trans[:, 3])
    return X_trans[:, :3] / X_trans[:, 3:]

def apply_homography_batch(H_batch: torch.Tensor, X: torch.Tensor) -> torch.Tensor:
    """
    Efficiently apply batched 4x4 homographies to 3D points.
    
    Args:
        H_batch: Tensor of shape (B, 4, 4)
        X:       Tensor of shape (N, 3)
    Returns:
        Transformed points: Tensor of shape (B, N, 3)
    """
    B = H_batch.shape[0]
    N = X.shape[0]
    
    # Append 1 to each point: (N, 4)
    ones = torch.ones((N, 1), dtype=X.dtype, device=X.device)
    X_h = torch.cat([X, ones], dim=1)  # (N, 4)

    # Apply homographies: (B, 4, 4) x (N, 4)^T → (B, 4, N)
    X_h = X_h.T.unsqueeze(0).expand(B, 4, N)  # (B, 4, N)
    X_trans = torch.bmm(H_batch, X_h)  # (B, 4, N)

    # Perspective divide
    X_trans = X_trans[:, :3, :] / X_trans[:, 3:4, :]  # (B, 3, N)
    
    # Transpose to (B, N, 3)
    return X_trans.permute(0, 2, 1)

def estimate_3D_homography(X_src_batch, X_dst_batch):
    """
    Estimate batch of 3D Homography.
    
    Inputs:
        X_src_batch: (B, N, 3)
        X_dst_batch: (B, N, 3)
        
    Returns:
        H_batch: (B, 4, 4)
    """
    B, N, _ = X_src_batch.shape
    ones = np.ones((B, N))

    x, y, z = X_src_batch[:, :, 0], X_src_batch[:, :, 1], X_src_batch[:, :, 2]
    xp, yp, zp = X_dst_batch[:, :, 0], X_dst_batch[:, :, 1], X_dst_batch[:, :, 2]

    # Prepare matrices
    A = np.zeros((B, 3 * N, 16))

    stacked_X = np.stack([x, y, z, ones], axis=2)  # (B, N, 4)

    # Fill in A
    A[:, 0::3, 0:4] = -stacked_X
    A[:, 0::3, 12:16] = np.stack([x * xp, y * xp, z * xp, xp], axis=2)

    A[:, 1::3, 4:8] = -stacked_X
    A[:, 1::3, 12:16] = np.stack([x * yp, y * yp, z * yp, yp], axis=2)

    A[:, 2::3, 8:12] = -stacked_X
    A[:, 2::3, 12:16] = np.stack([x * zp, y * zp, z * zp, zp], axis=2)

    # Solve using null space
    H_batch = np.zeros((B, 4, 4))
    for i in range(B):
        nullvec = null_space(A[i])
        if nullvec.shape[1] == 0:
            H_batch[i] = np.eye(4)
            continue

        H = nullvec[:, 0].reshape(4, 4)
        if H[3, 3] == 0:
            H_batch[i] = np.eye(4)
            continue

        H = H / H[3, 3]

        det = np.linalg.det(H)
        if np.isnan(det) or det < 0.0001:
            H_batch[i] = np.eye(4)
        else:
            H_batch[i] = H / det**0.25

    return torch.tensor(H_batch, dtype = torch.float32, device='cuda')

def is_planar(X, threshold=5e-2):
    X_centered = X - X.mean(axis=0)
    _, S, _ = np.linalg.svd(X_centered)
    normal_strength = S[-1] / S[0]
    return normal_strength < threshold

def scale(X):
    centroid = X.mean(axis=0)
    X_centered = X - centroid  # move centroid to origin

    # Compute average distance to the origin after centering
    avg_norm = np.linalg.norm(X_centered, axis=1).mean()

    # Desired average distance is sqrt(3)
    desired_avg_norm = np.sqrt(3)

    # Compute the uniform scaling factor
    scale = desired_avg_norm / avg_norm

    # Construct the 4x4 similarity transform matrix
    T = np.eye(4)
    T[:3, :3] *= scale  # apply scaling
    T[:3, 3] = -scale * centroid  # apply translation

    X_h = np.hstack([X, np.ones((X.shape[0], 1))])  # shape: (N, 4)

    # Step 2: Apply the transform
    X_transformed_h = (T @ X_h.T).T  # shape: (N, 4)

    # Step 3: Convert back to 3D (drop the homogeneous coordinate)
    X_transformed = X_transformed_h[:, :3]

    return T, X_transformed

def ransac_projective(X1_np, X2_np, threshold=0.01, max_iter=300, sample_size=5):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Convert to torch tensors on GPU
    X1 = torch.tensor(X1_np, dtype=torch.float32, device=device)
    X2 = torch.tensor(X2_np, dtype=torch.float32, device=device)
    N = X1.shape[0]

    # Sample indices for each hypothesis.
    indices = torch.randint(0, N, (max_iter, sample_size), device=device)

    # Gather sampled point sets.
    X1_samples = torch.stack([X1[idx] for idx in indices])  # (max_iter, sample_size, 3)
    X2_samples = torch.stack([X2[idx] for idx in indices])  # (max_iter, sample_size, 3)

    # Estimate homographies.
    H_ests = estimate_3D_homography(X1_samples.cpu().numpy(), X2_samples.cpu().numpy())

    # Apply homographies to all points.
    X2_preds = apply_homography_batch(H_ests, X1)

    # Compute Euclidean error.
    errors = torch.norm(X2_preds - X2[None, :, :], dim=2)

    # Compute inlier masks and counts.
    inlier_masks = errors < threshold  # (max_iter, N)
    inlier_counts = inlier_masks.sum(dim=1)

    # Select best hypothesis
    best_idx = torch.argmax(inlier_counts)
    best_H = H_ests[best_idx].cpu().numpy()

    return best_H
```

### vggt_slam/loop_closure.py

```python
import os
import heapq
from pathlib import Path as _Path
from typing import NamedTuple, List, Tuple

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image

from salad.eval import load_model  # SALAD


def _patch_torch_hub_offline():
    """
    Force torch.hub.load() to use local cached repos for offline environments.
    Looks for:
      - <hub_dir>/serizba_salad_main
      - <hub_dir>/facebookresearch_dinov2_main
    """
    import torch as _torch

    hub_dir = _Path(_torch.hub.get_dir())
    mapping = {}
    salad = hub_dir / "serizba_salad_main"
    dinov2 = hub_dir / "facebookresearch_dinov2_main"
    if salad.exists():
        mapping["serizba/salad"] = salad
    if dinov2.exists():
        mapping["facebookresearch/dinov2"] = dinov2

    if not mapping:
        return {}

    _orig = _torch.hub.load

    def _load(repo_or_dir, model, *args, **kwargs):
        if repo_or_dir in mapping:
            kwargs["source"] = "local"
            repo_or_dir = str(mapping[repo_or_dir])
        return _orig(repo_or_dir, model, *args, **kwargs)

    _torch.hub.load = _load
    return mapping


device = "cuda" if torch.cuda.is_available() else "cpu"

tensor_to_pil = T.ToPILImage()

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


def input_transform(image_size=None):
    transform_list = [T.ToTensor(), T.Normalize(mean=MEAN, std=STD)]
    if image_size:
        transform_list.insert(0, T.Resize(image_size, interpolation=T.InterpolationMode.BILINEAR))
    return T.Compose(transform_list)


class LoopMatch(NamedTuple):
    score: float
    query_submap_id: int
    query_submap_frame: int
    detected_submap_id: int
    detected_submap_frame: int


class LoopMatchQueue:
    def __init__(self, max_size: int):
        self.max_size = int(max_size)
        self.heap: List[Tuple[float, LoopMatch]] = []

    def add(self, match: LoopMatch):
        # heap stores (-score, match) so that "largest negative" == smallest score is at end after sort
        item = (-float(match.score), match)
        if len(self.heap) < self.max_size:
            heapq.heappush(self.heap, item)
        else:
            heapq.heappushpop(self.heap, item)

    def get_matches(self):
        # return sorted by score ascending (best first)
        return [m for _, m in sorted(self.heap, reverse=True)]


class ImageRetrieval:
    def __init__(self, input_size=224):
        _patch_torch_hub_offline()

        # Ensure salad repo is present in hub cache (offline-safe with patch)
        try:
            torch.hub.load("serizba/salad", "dinov2_salad")
        except Exception:
            # ignore; we will load checkpoint below anyway
            pass

        ckpt_pth = os.path.join(torch.hub.get_dir(), "checkpoints/dino_salad.ckpt")
        self.model = load_model(ckpt_pth)
        self.model.eval()
        self.model.to(device)

        self.transform = input_transform((input_size, input_size))

    @torch.no_grad()
    def get_batch_descriptors(self, imgs):
        """
        imgs: torch.Tensor (B,3,H,W) OR list of torch.Tensor/PIL images.
        return: torch.Tensor (B,D) on CPU.
        """
        # to list of PIL
        pil_list = []
        if isinstance(imgs, torch.Tensor):
            # (B,3,H,W)
            for i in range(imgs.shape[0]):
                im = imgs[i].detach().cpu()
                # if normalized already, just convert; else assume 0..1
                pil_list.append(tensor_to_pil(im))
        else:
            for im in imgs:
                if isinstance(im, Image.Image):
                    pil_list.append(im)
                elif isinstance(im, torch.Tensor):
                    pil_list.append(tensor_to_pil(im.detach().cpu()))
                else:
                    # numpy HWC BGR/RGB
                    arr = np.asarray(im)
                    if arr.ndim == 3 and arr.shape[2] == 3:
                        pil_list.append(Image.fromarray(arr.astype(np.uint8)))
                    else:
                        raise ValueError("Unsupported image type for retrieval")

        batch = torch.stack([self.transform(im) for im in pil_list], dim=0).to(device)

        out = self.model(batch)

        # robust unpack
        if isinstance(out, dict):
            if "global_descriptor" in out:
                desc = out["global_descriptor"]
            elif "descriptor" in out:
                desc = out["descriptor"]
            else:
                desc = next(iter(out.values()))
        elif isinstance(out, (tuple, list)):
            desc = out[0]
        else:
            desc = out

        desc = desc.detach()
        # normalize shape -> (B,D)
        if desc.dim() == 1:
            desc = desc.unsqueeze(0)
        elif desc.dim() == 3:
            # common: (B,1,D) or (1,B,D)
            if desc.shape[1] == 1:
                desc = desc.squeeze(1)
            elif desc.shape[0] == 1:
                desc = desc.squeeze(0)
            else:
                desc = desc.reshape(desc.shape[0], -1)
        elif desc.dim() > 3:
            desc = desc.reshape(desc.shape[0], -1)

        desc = desc.float()
        # optional normalize (helps dot-product / euclid stability)
        desc = torch.nn.functional.normalize(desc, dim=-1)

        return desc.cpu()

    def get_all_submap_embeddings(self, submap):
        frames = submap.get_all_frames()
        return self.get_batch_descriptors(frames)

    def find_loop_closures(
        self,
        map,
        submap,
        max_similarity_thres: float = 0.8,
        max_loop_closures: int = 0,
    ):
        """
        NOTE:
          - best_score is distance (smaller better) if map uses L2.
          - keep condition: best_score < max_similarity_thres
        """
        if max_loop_closures <= 0:
            return []

        matches_queue = LoopMatchQueue(max_loop_closures)

        query_id = 0
        for query_vector in submap.get_all_retrieval_vectors():
            try:
                qv = query_vector
                if isinstance(qv, torch.Tensor):
                    qv = qv.detach().cpu().float().reshape(-1)
                else:
                    qv = torch.as_tensor(qv).float().reshape(-1)

                best_score, best_submap_id, best_frame_id = map.retrieve_best_score_frame(
                    qv,
                    submap.get_id(),
                    ignore_last_submap=True,
                )

                if best_score < float(max_similarity_thres):
                    matches_queue.add(
                        LoopMatch(
                            float(best_score),
                            int(submap.get_id()),
                            int(query_id),
                            int(best_submap_id),
                            int(best_frame_id),
                        )
                    )

            except Exception as e:
                # 关键兜底：任何 embedding 维度/类型异常，不要炸整个 SLAM
                # 直接跳过这个 query frame
                # print(f"[WARN][loop_closure] skip query {query_id}: {e}")
                pass

            query_id += 1

        return matches_queue.get_matches()


def is_point_in_fov(K, T_wc, point_world, image_size, fov_padding=0.0):
    """
    Check if a 3D point is inside the camera frustum defined by K and T_wc.
    """
    T_cw = np.linalg.inv(T_wc)  # World to camera
    point_cam = T_cw[:3, :3] @ point_world + T_cw[:3, 3]
    if point_cam[2] <= 1e-8:
        return False

    x = K[0, 0] * (point_cam[0] / point_cam[2]) + K[0, 2]
    y = K[1, 1] * (point_cam[1] / point_cam[2]) + K[1, 2]

    W, H = image_size[0], image_size[1]
    pad = float(fov_padding)
    return (-pad <= x <= W + pad) and (-pad <= y <= H + pad)


def frustums_overlap(K1, T1, K2, T2, image_size):
    """
    Approximate overlap check by sampling points on near plane of one frustum and checking in other.
    """
    # sample a few points in image plane for camera1
    W, H = image_size[0], image_size[1]
    pts = [
        np.array([0.0, 0.0, 1.0]),
        np.array([W, 0.0, 1.0]),
        np.array([0.0, H, 1.0]),
        np.array([W, H, 1.0]),
        np.array([W / 2.0, H / 2.0, 1.0]),
    ]

    K1_inv = np.linalg.inv(K1)
    # convert to world points (arbitrary depth=1)
    for p in pts:
        ray = K1_inv @ p
        pw = (T1[:3, :3] @ ray) + T1[:3, 3]
        if is_point_in_fov(K2, T2, pw, image_size, fov_padding=0.0):
            return True
    return False

```

### vggt_slam/map.py

```python
import os
import numpy as np
import torch
import open3d as o3d
from scipy.spatial.transform import Rotation as R

class GraphMap:
    def __init__(self):
        self.submaps = dict()
    
    def get_num_submaps(self):
        return len(self.submaps)

    def add_submap(self, submap):
        submap_id = submap.get_id()
        self.submaps[submap_id] = submap
    
    def get_largest_key(self):
        if len(self.submaps) == 0:
            return -1
        return max(self.submaps.keys())
    
    def get_submap(self, id):
        return self.submaps[id]

    def get_latest_submap(self):
        return self.get_submap(self.get_largest_key())
    
    def retrieve_best_score_frame(self, query_vector, current_submap_id, ignore_last_submap=True):
        overall_best_score = 1000
        overall_best_submap_id = 0
        overall_best_frame_index = 0
        # search for best image to target image
        for submap_key in self.submaps.keys():
            if submap_key == current_submap_id:
                continue

            if ignore_last_submap and (submap_key == current_submap_id-1):
                continue

            else:
                submap = self.submaps[submap_key]
                submap_embeddings = submap.get_all_retrieval_vectors()
                scores = []
                for embedding in submap_embeddings:
                    score = torch.linalg.norm(embedding-query_vector)
                    scores.append(score.item())
                
                best_score_id = np.argmin(scores)
                best_score = scores[best_score_id]

                if best_score < overall_best_score:
                    overall_best_score = best_score
                    overall_best_submap_id = submap_key
                    overall_best_frame_index = best_score_id

        return overall_best_score, overall_best_submap_id, overall_best_frame_index

    def get_frames_from_loops(self, loops):
        frames = []
        for detected_loop in loops:
            frames.append(self.submaps[detected_loop.detected_submap_id].get_frame_at_index(detected_loop.detected_submap_frame))
        
        return frames
    
    def update_submap_homographies(self, graph):

        for submap in self.get_submaps():
            submap_key = submap.get_id()
            H = graph.get_homography(submap_key)

            if H is None:
                continue

        # gtsam object -> matrix
            if hasattr(H, "matrix"):
                H = H.matrix()

        # ensure numpy 4x4
            H = np.asarray(H)
            if H.shape != (4, 4):
                print(f"[WARN] invalid homography shape for submap {submap_key}: {H.shape}")
                continue

            submap.set_reference_homography(H)

    
    def get_submaps(self):
        return self.submaps.values()

    def ordered_submaps_by_key(self):
        for k in sorted(self.submaps):
            yield self.submaps[k]

    def write_poses_to_file(self, file_name):
        with open(file_name, "w") as f:
            for submap in self.ordered_submaps_by_key():
                poses = submap.get_all_poses_world(ignore_loop_closure_frames=True)
                frame_ids = submap.get_frame_ids()
                assert len(poses) == len(frame_ids), "Number of provided poses and number of frame ids do not match"
                for frame_id, pose in zip(frame_ids, poses):
                    x, y, z = pose[0:3, 3]
                    rotation_matrix = pose[0:3, 0:3]
                    quaternion = R.from_matrix(rotation_matrix).as_quat() # x, y, z, w
                    output = np.array([float(frame_id), x, y, z, *quaternion])
                    f.write(" ".join(f"{v:.8f}" for v in output) + "\n")

    def save_framewise_pointclouds(self, file_name):
        os.makedirs(file_name, exist_ok=True)
        for submap in self.ordered_submaps_by_key():
            pointclouds, frame_ids, conf_masks = submap.get_points_list_in_world_frame(ignore_loop_closure_frames=True)
            for frame_id, pointcloud, conf_masks in zip(frame_ids, pointclouds, conf_masks):
                # save pcd as numpy array
                np.savez(f"{file_name}/{frame_id}.npz", pointcloud=pointcloud, mask=conf_masks)
                

    def write_points_to_file(self, file_name):
        pcd_all = []
        colors_all = []
        for submap in self.ordered_submaps_by_key():
            pcd = submap.get_points_in_world_frame()
            pcd = pcd.reshape(-1, 3)
            pcd_all.append(pcd)
            colors_all.append(submap.get_points_colors())
        pcd_all = np.concatenate(pcd_all, axis=0)
        colors_all = np.concatenate(colors_all, axis=0)
        if colors_all.max() > 1.0:
            colors_all = colors_all / 255.0
        pcd_all = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pcd_all))
        pcd_all.colors = o3d.utility.Vector3dVector(colors_all)
        o3d.io.write_point_cloud(file_name, pcd_all)
```

### vggt_slam/semantic_backend.py

```python
import os
import sys
import cv2
import numpy as np
from dataclasses import dataclass
from pathlib import Path
from collections import OrderedDict

# Optional deps
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_OK = True
except Exception:
    torch = None
    nn = None
    F = None
    TORCH_OK = False

try:
    from PIL import Image
    PIL_OK = True
except Exception:
    Image = None
    PIL_OK = False


@dataclass
class SemanticBackendCfg:
    # Deep backend (DINOv2 + metric head)
    use_deep: bool = True
    dinov2_repo: str = ""           # e.g. /path/to/dinov2
    dinov2_name: str = "dinov2_vits14"
    dinov2_ckpt: str = ""          # optional (some repos auto-download)
    metric_head_ckpt: str = ""     # path to .pth produced by your training script
    deep_img_size: int = 224

    # HOG fallback
    hog_img_size: int = 128
    hog_winsize: int = 128
    hog_blocksize: int = 16
    hog_blockstride: int = 8
    hog_cellsize: int = 8
    hog_nbins: int = 9

    # Cache
    cache_size: int = 4096


if TORCH_OK:
    class MetricHead(nn.Module):
        """Simple MLP metric head."""

        def __init__(self, dim: int = 384, out_dim: int = 128):
            super().__init__()
            self.fc1 = nn.Linear(dim, dim)
            self.fc2 = nn.Linear(dim, out_dim)

        def forward(self, x):
            x = F.relu(self.fc1(x))
            x = self.fc2(x)
            return x
else:
    class MetricHead(object):
        def __init__(self, *args, **kwargs):
            raise RuntimeError("torch is required for MetricHead")

        def forward(self, x):
            raise RuntimeError("torch is required for MetricHead")


class SemanticBackend:
    """Semantic backend for gating and reweighting edges.

    - If deep backend is available (DINOv2 + metric head), uses deep embeddings.
    - Otherwise falls back to HOG features.

    IMPORTANT: `similarity(a, b)` accepts:
      - file path (str/Path)
      - numpy image (H,W,3) or (H,W)
      - torch tensor (C,H,W) or (H,W,C)
      - PIL Image
    """

    def __init__(self, cfg_path: str = ""):
        self.cfg = SemanticBackendCfg()
        self.device = "cuda" if (TORCH_OK and torch.cuda.is_available()) else "cpu"

        # cache for path-based embeddings
        self._cache = OrderedDict()  # key -> np.ndarray

        # Try read cfg_path (optional)
        self._apply_cfg_path(cfg_path)

        self.use_deep = bool(self.cfg.use_deep and TORCH_OK)
        self.deep_model = None
        self.metric_head = None

        if self.use_deep:
            ok = self._init_deep()
            if ok:
                print(f"[SemanticBackend] deep enabled (dinov2+metric_head) on {self.device}")
            else:
                self.use_deep = False
                print("[SemanticBackend] deep disabled (missing repo/ckpt), fallback to HOG")
        else:
            if not TORCH_OK:
                print("[SemanticBackend] deep disabled (missing torch), fallback to HOG")
            else:
                print("[SemanticBackend] deep disabled (cfg.use_deep=False), fallback to HOG")

        # HOG descriptor (always available as fallback)
        ws = (int(self.cfg.hog_winsize), int(self.cfg.hog_winsize))
        bs = (int(self.cfg.hog_blocksize), int(self.cfg.hog_blocksize))
        bstr = (int(self.cfg.hog_blockstride), int(self.cfg.hog_blockstride))
        cs = (int(self.cfg.hog_cellsize), int(self.cfg.hog_cellsize))
        self.hog = cv2.HOGDescriptor(ws, bs, bstr, cs, int(self.cfg.hog_nbins))

    # -------------------------
    # Config helpers
    # -------------------------

    def _apply_cfg_path(self, cfg_path: str):
        if not cfg_path:
            return
        p = Path(cfg_path)
        if not p.exists():
            # Allow passing a directory containing semantic_backend.yaml
            if p.is_dir():
                cand = p / "semantic_backend.yaml"
                if cand.exists():
                    p = cand
                else:
                    return
            else:
                return
        try:
            import yaml
            cfg_dict = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
        except Exception:
            return

        # shallow merge
        for k, v in cfg_dict.items():
            if hasattr(self.cfg, k):
                setattr(self.cfg, k, v)

    def _init_deep(self) -> bool:
        # Require dinov2 repo and metric head ckpt
        repo = str(self.cfg.dinov2_repo or "").strip()
        head_ckpt = str(self.cfg.metric_head_ckpt or "").strip()
        if (not repo) or (not Path(repo).exists()) or (not head_ckpt) or (not Path(head_ckpt).exists()):
            return False

        # Make repo importable
        if repo not in sys.path:
            sys.path.insert(0, repo)

        # DINOv2 loading strategies (best-effort)
        try:
            # Common pattern in dinov2 repo
            # e.g. from dinov2.hub.backbones import dinov2_vits14
            from dinov2.hub.backbones import (  # type: ignore
                dinov2_vits14,
                dinov2_vitb14,
                dinov2_vitl14,
                dinov2_vitg14,
            )
            name = self.cfg.dinov2_name
            fn = {
                "dinov2_vits14": dinov2_vits14,
                "dinov2_vitb14": dinov2_vitb14,
                "dinov2_vitl14": dinov2_vitl14,
                "dinov2_vitg14": dinov2_vitg14,
            }.get(name, dinov2_vits14)
            self.deep_model = fn(pretrained=True)
        except Exception:
            # fallback: torch.hub (may fail offline)
            try:
                self.deep_model = torch.hub.load(repo, self.cfg.dinov2_name, source="local", pretrained=True)
            except Exception:
                return False

        self.deep_model.eval().to(self.device)

        # Load metric head
        ckpt = torch.load(head_ckpt, map_location="cpu")
        dim = int(ckpt.get("dim", 384))
        out_dim = int(ckpt.get("out_dim", 128))
        self.metric_head = MetricHead(dim=dim, out_dim=out_dim)
        state = ckpt.get("state_dict", ckpt)
        # tolerate checkpoints saved from DistributedDataParallel
        fixed_state = {k.replace("module.", ""): v for k, v in state.items()}
        self.metric_head.load_state_dict(fixed_state, strict=False)
        self.metric_head.eval().to(self.device)
        return True

    # -------------------------
    # Input conversion
    # -------------------------

    def _read_bgr(self, path: str) -> np.ndarray:
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError(f"cv2.imread failed: {path}")
        return img

    def _to_bgr(self, frame) -> np.ndarray:
        """Convert input to uint8 BGR (OpenCV order)."""
        # Robustly handle numpy scalar / size-1 arrays that accidentally contain a
        # file path (e.g. np.asarray('/path/to/img.jpg')). Without this, numpy may
        # treat it as dtype('<U..') and later arithmetic will crash.
        try:
            if isinstance(frame, np.ndarray) and getattr(frame, "dtype", None) is not None:
                if frame.dtype.kind in ("U", "S", "O"):
                    if frame.ndim == 0:
                        frame = str(frame.item())
                    elif frame.size == 1:
                        frame = str(frame.reshape(-1)[0])
        except Exception:
            # Best-effort only.
            pass
        # 1) path
        if isinstance(frame, (str, Path)):
            return self._read_bgr(str(frame))

        # 2) PIL
        if PIL_OK and isinstance(frame, Image.Image):
            rgb = np.array(frame.convert("RGB"), dtype=np.uint8)
            return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

        x = frame

        # 3) torch tensor
        if TORCH_OK and torch.is_tensor(x):
            x = x.detach().cpu().numpy()

        # 4) numpy
        x = np.asarray(x)

        # Handle grayscale
        if x.ndim == 2:
            x = np.stack([x, x, x], axis=-1)

        # Handle CHW
        if x.ndim == 3 and x.shape[0] == 3 and x.shape[-1] != 3:
            x = np.transpose(x, (1, 2, 0))

        if x.ndim != 3 or x.shape[-1] != 3:
            raise ValueError(f"unsupported frame shape: {x.shape}")

        # dtype normalization
        if x.dtype != np.uint8:
            x = x.astype(np.float32)
            mx = float(np.max(x)) if x.size else 0.0
            if mx > 1.5:
                x = np.clip(x, 0.0, 255.0)
            else:
                x = np.clip(x * 255.0, 0.0, 255.0)
            x = x.astype(np.uint8)

        # Assume input is RGB unless it clearly comes from cv2 (hard to detect).
        # Most tensors/arrays in this project are RGB (PIL/torch). Convert RGB->BGR.
        # If caller already provides BGR, similarity still works reasonably for HOG.
        try:
            x = cv2.cvtColor(x, cv2.COLOR_RGB2BGR)
        except Exception:
            pass
        return x

    # -------------------------
    # Embeddings
    # -------------------------

    def _resize_square(self, bgr: np.ndarray, size: int) -> np.ndarray:
        return cv2.resize(bgr, (size, size), interpolation=cv2.INTER_AREA)

    def _hog_embedding(self, bgr: np.ndarray) -> np.ndarray:
        bgr = self._resize_square(bgr, int(self.cfg.hog_img_size))
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        feat = self.hog.compute(gray)
        feat = feat.reshape(-1).astype(np.float32)
        n = np.linalg.norm(feat) + 1e-12
        return feat / n

    def _deep_embedding(self, bgr: np.ndarray) -> np.ndarray:
        if not (self.use_deep and self.deep_model is not None and self.metric_head is not None):
            raise RuntimeError("deep backend not initialized")

        # bgr -> rgb
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        rgb = self._resize_square(rgb, int(self.cfg.deep_img_size))
        x = torch.from_numpy(rgb).float().to(self.device) / 255.0
        x = x.permute(2, 0, 1).unsqueeze(0)  # 1,3,H,W

        # ImageNet normalization (common default)
        mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1, 3, 1, 1)
        x = (x - mean) / std

        with torch.no_grad():
            # Many DINOv2 backbones return a dict or tuple; handle common cases.
            y = self.deep_model(x)
            if isinstance(y, dict):
                # prefer cls token / pooled output if present
                for k in ("x_norm_clstoken", "x_norm", "cls", "pooled"):
                    if k in y:
                        y = y[k]
                        break
            if isinstance(y, (list, tuple)):
                y = y[0]
            y = y.reshape(y.shape[0], -1)
            y = self.metric_head(y)
            y = F.normalize(y, dim=-1)
        return y.squeeze(0).detach().cpu().numpy().astype(np.float32)

    def embedding(self, frame) -> np.ndarray:
        # Cache only for path inputs
        cache_key = None
        if isinstance(frame, (str, Path)):
            cache_key = str(frame)
            if cache_key in self._cache:
                # LRU refresh
                v = self._cache.pop(cache_key)
                self._cache[cache_key] = v
                return v

        bgr = self._to_bgr(frame)
        if self.use_deep:
            emb = self._deep_embedding(bgr)
        else:
            emb = self._hog_embedding(bgr)

        if cache_key is not None:
            self._cache[cache_key] = emb
            if len(self._cache) > int(self.cfg.cache_size):
                self._cache.popitem(last=False)
        return emb

    # -------------------------
    # Public API
    # -------------------------

    def similarity(self, frame_a, frame_b) -> float:
        ea = self.embedding(frame_a)
        eb = self.embedding(frame_b)
        s = float(np.dot(ea, eb) / ((np.linalg.norm(ea) + 1e-12) * (np.linalg.norm(eb) + 1e-12)))
        s = max(-1.0, min(1.0, s))
        # HOG similarity can be slightly negative; map to [0,1] to be consistent
        if not self.use_deep:
            s = 0.5 * (s + 1.0)
        return float(s)

```

### vggt_slam/slam_utils.py

```python
import os
import re

def slice_with_overlap(lst, n, k):
    if n <= 0 or k < 0:
        raise ValueError("n must be greater than 0 and k must be non-negative")
    result = []
    i = 0
    while i < len(lst):
        result.append(lst[i:i + n])
        i += max(1, n - k)  # Ensure progress even if k >= n
    return result


def sort_images_by_number(image_paths):
    def extract_number(path):
        filename = os.path.basename(path)
        # Match decimal or integer number in filename
        match = re.search(r'\d+(?:\.\d+)?', filename)
        return float(match.group()) if match else float('inf')

    return sorted(image_paths, key=extract_number)

def downsample_images(image_names, downsample_factor):
    """
    Downsamples a list of image names by keeping every `downsample_factor`-th image.
    
    Args:
        image_names (list of str): List of image filenames.
        downsample_factor (int): Factor to downsample the list. E.g., 2 keeps every other image.

    Returns:
        list of str: Downsampled list of image filenames.
    """
    return image_names[::downsample_factor]
```

### vggt_slam/solver.py

```python
# -*- coding: utf-8 -*-
from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import cv2
import gtsam
import matplotlib.pyplot as plt
import torch
import open3d as o3d
import viser
import viser.transforms as viser_tf
from termcolor import colored

from vggt.utils.geometry import closed_form_inverse_se3, unproject_depth_map_to_point_map
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.pose_enc import pose_encoding_to_extri_intri

from vggt_slam.loop_closure import ImageRetrieval
from vggt_slam.frame_overlap import FrameTracker
from vggt_slam.map import GraphMap
from vggt_slam.submap import Submap
from vggt_slam.h_solve import ransac_projective
from vggt_slam.gradio_viewer import TrimeshViewer

def color_point_cloud_by_confidence(pcd, confidence, cmap="viridis"):
    """
    Color a point cloud based on per-point confidence values.
    Parameters:
        pcd (o3d.geometry.PointCloud): The point cloud.
        confidence (np.ndarray): Confidence values, shape (N,).
        cmap (str): Matplotlib colormap name.
    """
    assert len(confidence) == len(pcd.points), "Confidence length must match number of points"
    confidence_normalized = (confidence - np.min(confidence)) / (np.ptp(confidence) + 1e-8)
    colormap = plt.get_cmap(cmap)
    colors = colormap(confidence_normalized)[:, :3]
    pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd

class Viewer:
    """纯可视化：不要在这里初始化 semantic backend（避免重复加载/副作用）"""

    def __init__(self, port: int = 8080):
        print(f"Starting viser server on port {port}")
        self.server = viser.ViserServer(host="0.0.0.0", port=port)
        self.server.gui.configure_theme(titlebar_content=None, control_layout="collapsible")

        self.gui_show_frames = self.server.gui.add_checkbox("Show Cameras", initial_value=True)
        self.gui_show_frames.on_update(self._on_update_show_frames)

        self.submap_frames: Dict[int, List[viser.FrameHandle]] = {}
        self.submap_frustums: Dict[int, List[viser.CameraFrustumHandle]] = {}

        num_rand_colors = 250
        self.random_colors = np.random.randint(0, 256, size=(num_rand_colors, 3), dtype=np.uint8)

    def visualize_frames(self, extrinsics: np.ndarray, images_: np.ndarray, submap_id: int, image_scale: float = 0.5) -> None:
        """
        Add camera frames and frustums to the scene for a specific submap.
        extrinsics: (S, 3, 4)
        images_:    (S, 3, H, W)
        """
        if isinstance(images_, torch.Tensor):
            images_ = images_.cpu().numpy()

        if submap_id not in self.submap_frames:
            self.submap_frames[submap_id] = []
            self.submap_frustums[submap_id] = []

        S = extrinsics.shape[0]
        for img_id in range(S):
            cam2world_3x4 = extrinsics[img_id]
            T_world_camera = viser_tf.SE3.from_matrix(cam2world_3x4)

            frame_name = f"submap_{submap_id}/frame_{img_id}"
            frustum_name = f"{frame_name}/frustum"

            frame_axis = self.server.scene.add_frame(
                frame_name,
                wxyz=T_world_camera.rotation().wxyz,
                position=T_world_camera.translation(),
                axes_length=0.05,
                axes_radius=0.002,
                origin_radius=0.002,
            )
            frame_axis.visible = self.gui_show_frames.value
            self.submap_frames[submap_id].append(frame_axis)

            img = images_[img_id]
            img = (img.transpose(1, 2, 0) * 255).astype(np.uint8)

            h, w = img.shape[:2]
            fy = 1.1 * h
            fov = 2 * np.arctan2(h / 2, fy)

            img_resized = cv2.resize(
                img,
                (int(img.shape[1] * image_scale), int(img.shape[0] * image_scale)),
                interpolation=cv2.INTER_AREA,
            )

            frustum = self.server.scene.add_camera_frustum(
                frustum_name,
                fov=fov,
                aspect=w / h,
                scale=0.05,
                image=img_resized,
                line_width=3.0,
                color=self.random_colors[submap_id],
            )
            frustum.visible = self.gui_show_frames.value
            self.submap_frustums[submap_id].append(frustum)

    def _on_update_show_frames(self, _) -> None:
        visible = self.gui_show_frames.value
        for frames in self.submap_frames.values():
            for f in frames:
                f.visible = visible
        for frustums in self.submap_frustums.values():
            for fr in frustums:
                fr.visible = visible

@dataclass
class EdgeStat:
    """记录每条图边（里程计/回环）的语义相似度与权重，便于论文做 ablation & 可解释性。"""
    time: str
    edge_type: str  # odom|loop
    src: int
    dst: int
    sim: float
    w: float
    n_good: int
    mask_thr: float
    mask_fallback_or: int
    weight_mode: str
    # --- corridor anti-aliasing additions ---
    margin: float = 0.0   # top1-top2 similarity gap (retrieval space)
    u: float = 1.0        # uniqueness factor in [u_min, 1]

class Solver:
    def __init__(
        self,
        init_conf_threshold: float,
        use_point_map: bool = False,
        visualize_global_map: bool = False,
        use_sim3: bool = False,
        gradio_mode: bool = False,
        vis_stride: int = 1,
        vis_point_size: float = 0.001,
        # --- semantic backend ---
        use_semantic_backend: bool = False,
        semantic_backend_cfg: str = "",
        semantic_min_sim: float = 0.25,
        # --- semantic gate（筛选 loop candidate / retrieved frames）---
        semantic_gate_mode: str = "both",   # off|filter|retrieved|both
        disable_semantic_gate: bool = False,
        # --- semantic factor reweighting（核心：语义参与图优化）---
        semantic_weight_mode: str = "loop_only",  # off|loop_only|all_edges
        semantic_w_min: float = 0.25,
        semantic_w_max: float = 4.0,
        semantic_w_gamma: float = 2.0,
        semantic_w_s0: float = 0.25,
        semantic_w_degen_beta: float = 0.0,
        semantic_w_degen_ref_good: int = 2000,
        # --- corridor anti-aliasing (uniqueness) ---
        semantic_u_enable: bool = True,
        semantic_u_topk_submaps: int = 8,
        semantic_u_m0: float = 0.05,
        semantic_u_min: float = 0.25,
        # --- loop ambiguity control (within retrieved loop candidates) ---
        # Keep only top-K candidates after semantic scoring; additionally compute
        # a margin (best-second best) and down-weight ambiguous loops.
        semantic_loop_topk: int = 1,
        semantic_loop_margin_thr: float = 0.02,
        # --- stats logging ---
        edge_stats_path: str = "",
        **kwargs,
    ):
        # semantic backend
        self.use_semantic_backend = bool(use_semantic_backend)
        self.semantic_backend_cfg = semantic_backend_cfg
        self.semantic_min_sim = float(semantic_min_sim)
        self.semantic_backend = None
        if self.use_semantic_backend:
            try:
                from vggt_slam.semantic_backend import SemanticBackend
                self.semantic_backend = SemanticBackend(cfg_path=self.semantic_backend_cfg)
            except Exception as e:
                print(f"[WARN] semantic backend init failed -> disabled: {e}")
                self.use_semantic_backend = False
                self.semantic_backend = None

        # semantic gate
        self.semantic_gate_mode = str(semantic_gate_mode)
        if self.semantic_gate_mode not in ("off", "filter", "retrieved", "both"):
            print(f"[WARN] unknown semantic_gate_mode={self.semantic_gate_mode}, fallback to both")
            self.semantic_gate_mode = "both"
        self.disable_semantic_gate = bool(disable_semantic_gate)
        if (not self.use_semantic_backend) or (self.semantic_backend is None) or self.disable_semantic_gate:
            self.semantic_gate_mode = "off"

        # loop ambiguity control (only meaningful if semantic backend is enabled)
        self.semantic_loop_topk = int(semantic_loop_topk)
        self.semantic_loop_margin_thr = float(semantic_loop_margin_thr)

        # semantic weights
        self.semantic_weight_mode = str(semantic_weight_mode)
        if self.semantic_weight_mode not in ("off", "loop_only", "all_edges"):
            print(f"[WARN] unknown semantic_weight_mode={self.semantic_weight_mode}, fallback to off")
            self.semantic_weight_mode = "off"

        self.semantic_w_min = float(semantic_w_min)
        self.semantic_w_max = float(semantic_w_max)
        self.semantic_w_gamma = float(semantic_w_gamma)
        self.semantic_w_s0 = float(semantic_w_s0)
        self.semantic_w_degen_beta = float(semantic_w_degen_beta)
        self.semantic_w_degen_ref_good = int(semantic_w_degen_ref_good)

        if (not self.use_semantic_backend) or (self.semantic_backend is None):
            # 没有语义后端就不要开权重（避免 baseline 行为被不小心改变）
            self.semantic_weight_mode = "off"

        # corridor anti-aliasing (uniqueness)
        self.semantic_u_enable = bool(semantic_u_enable)
        self.semantic_u_topk_submaps = int(semantic_u_topk_submaps)
        self.semantic_u_m0 = float(semantic_u_m0)
        self.semantic_u_min = float(semantic_u_min)

        # edge stats
        self.edge_stats_path = str(edge_stats_path).strip()
        self._edge_stats_fp = None
        self._edge_stats_writer = None
        self._edge_stats_init_if_needed()

        self.init_conf_threshold = init_conf_threshold
        self.use_point_map = use_point_map
        self.gradio_mode = gradio_mode

        if self.gradio_mode:
            self.viewer = TrimeshViewer()
        else:
            self.viewer = Viewer()

        self.flow_tracker = FrameTracker()
        self.map = GraphMap()
        self.use_sim3 = use_sim3

        if self.use_sim3:
            from vggt_slam.graph_se3 import PoseGraph
        else:
            from vggt_slam.graph import PoseGraph
        self.graph = PoseGraph()

        self.image_retrieval = ImageRetrieval()
        self.current_working_submap = None

        self.first_edge = True
        self.T_w_kf_minus = None

        self.prior_pcd = None
        self.prior_conf = None

        self.vis_stride = vis_stride
        self.vis_point_size = vis_point_size

        print("Starting viser server...")

    # -------------------------
    # stats logging
    # -------------------------
    def _set_submap_attr(self, submap, attr_name: str, value):
        # Compatible setter for different Submap method names.
        candidates = [
            f'set_all_{attr_name}',
            f'add_all_{attr_name}',
            f'set_{attr_name}',
            f'add_{attr_name}',
        ]
        for m in candidates:
            fn = getattr(submap, m, None)
            if callable(fn):
                fn(value)
                return
        setattr(submap, attr_name, value)

    def _edge_stats_init_if_needed(self):
        if not self.edge_stats_path:
            return
        p = Path(self.edge_stats_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        is_new = (not p.exists())
        self._edge_stats_fp = p.open("a", newline="", encoding="utf-8")
        self._edge_stats_writer = csv.DictWriter(
            self._edge_stats_fp,
            fieldnames=[
                "time", "edge_type", "src", "dst", "sim", "w",
                "n_good", "mask_thr", "mask_fallback_or", "weight_mode",
                "margin", "u",
            ],
        )
        if is_new:
            self._edge_stats_writer.writeheader()
            self._edge_stats_fp.flush()

    def _edge_stats_log(self, stat: EdgeStat):
        if self._edge_stats_writer is None:
            return
        self._edge_stats_writer.writerow({
            "time": stat.time,
            "edge_type": stat.edge_type,
            "src": int(stat.src),
            "dst": int(stat.dst),
            "sim": f"{stat.sim:.6f}",
            "w": f"{stat.w:.6f}",
            "n_good": int(stat.n_good),
            "mask_thr": f"{stat.mask_thr:.6f}",
            "mask_fallback_or": int(stat.mask_fallback_or),
            "weight_mode": str(stat.weight_mode),
            "margin": f"{float(stat.margin):.6f}",
            "u": f"{float(stat.u):.6f}",
        })
        self._edge_stats_fp.flush()

    # -------------------------
    # semantic weight helpers
    # -------------------------
    def _semantic_sim_to_weight(self, sim: float) -> float:
        """
        sim∈[0,1] -> w
        约定：w=1 表示不改变；w>1 表示更相信（噪声更小）；w<1 表示更不相信（噪声更大）
        在 graph.py 内部建议按 sigma' = sigma0 / sqrt(w) 方式实现。
        """
        try:
            s = float(sim)
        except Exception:
            return 1.0
        if not np.isfinite(s):
            return 1.0
        s = max(0.0, min(1.0, s))

        s0 = max(0.0, min(0.999, float(self.semantic_w_s0)))
        if s <= s0:
            t = 0.0
        else:
            t = (s - s0) / (1.0 - s0)
        t = max(0.0, min(1.0, t))

        gamma = max(1e-6, float(self.semantic_w_gamma))
        wmin = float(self.semantic_w_min)
        wmax = float(self.semantic_w_max)
        if wmax < wmin:
            wmin, wmax = wmax, wmin

        w = wmin + (wmax - wmin) * (t ** gamma)
        if (not np.isfinite(w)) or w <= 0:
            return 1.0
        return float(w)

    def _apply_degeneracy_boost(self, w: float, n_good: int) -> float:
        """
        退化增强（可选）：
        d = 1 - clip(n_good/ref,0,1)
        w' = w*(1 + beta*d)
        """
        beta = float(self.semantic_w_degen_beta)
        if beta <= 0:
            return float(w)
        ref = float(max(1, int(self.semantic_w_degen_ref_good)))
        ng = float(max(0, int(n_good)))
        d = 1.0 - min(1.0, ng / ref)
        return float(w) * (1.0 + beta * d)

    # -------------------------
    # corridor anti-aliasing helpers (uniqueness)
    # -------------------------
    def _get_keyframe_retrieval_vec(self, submap: Submap, frame_idx: int) -> Optional[np.ndarray]:
        """取 retrieval embedding（归一化后）"""
        try:
            vecs = getattr(submap, "retrieval_vectors", None)
            if vecs is None:
                # 兼容不同实现：可能是 get_all_retrieval_vectors()
                if hasattr(submap, "get_all_retrieval_vectors"):
                    vecs = submap.get_all_retrieval_vectors()
            if vecs is None:
                return None
            v = np.asarray(vecs[int(frame_idx)], dtype=np.float32).reshape(-1)
            n = float(np.linalg.norm(v) + 1e-8)
            return v / n
        except Exception:
            return None

    def _compute_uniqueness_from_map(
        self,
        q_vec: Optional[np.ndarray],
        exclude_sid: int,
    ) -> Tuple[float, float, float, float]:
        """
        返回 (margin, u, top1, top2)
        margin = top1 - top2（越小越歧义）
        u ∈ [u_min, 1]（越小越降权）
        """
        if (not self.semantic_u_enable) or (q_vec is None):
            return 0.0, 1.0, 0.0, 0.0

        cands: List[np.ndarray] = []
        try:
            submaps = self.map.get_submaps()
            # 取最近若干个 submap 的关键帧 embedding
            for sm in submaps[-int(max(1, self.semantic_u_topk_submaps)):]:
                sid = int(sm.get_id())
                if sid == int(exclude_sid):
                    continue
                idx = int(sm.get_last_non_loop_frame_index())
                v = self._get_keyframe_retrieval_vec(sm, idx)
                if v is not None:
                    cands.append(v)
        except Exception:
            pass

        if len(cands) < 2:
            return 0.0, 1.0, 0.0, 0.0

        C = np.stack(cands, axis=0)  # (N,D) normalized
        sims = C @ q_vec             # cosine sims

        s_sorted = np.sort(sims)[::-1]
        top1 = float(s_sorted[0])
        top2 = float(s_sorted[1])
        margin = float(top1 - top2)

        m0 = float(max(1e-6, self.semantic_u_m0))
        u_min = float(np.clip(self.semantic_u_min, 0.0, 1.0))
        u = float(np.clip(margin / m0, u_min, 1.0))
        return margin, u, top1, top2

    # -------------------------
    # visualization helpers
    # -------------------------
    def set_point_cloud(self, points_in_world_frame, points_colors, name, point_size):
        if self.gradio_mode:
            self.viewer.add_point_cloud(points_in_world_frame, points_colors)
        else:
            self.viewer.server.scene.add_point_cloud(
                name="pcd_" + name,
                points=points_in_world_frame,
                colors=points_colors,
                point_size=point_size,
                point_shape="circle",
            )

    def set_submap_point_cloud(self, submap):
        points_in_world_frame = submap.get_points_in_world_frame(stride=self.vis_stride)
        points_colors = submap.get_points_colors(stride=self.vis_stride)
        name = str(submap.get_id())
        self.set_point_cloud(points_in_world_frame, points_colors, name, self.vis_point_size)

    def set_submap_poses(self, submap):
        extrinsics = submap.get_all_poses_world()
        if self.gradio_mode:
            for i in range(extrinsics.shape[0]):
                self.viewer.add_camera_pose(extrinsics[i])
        else:
            images = submap.get_all_frames()
            self.viewer.visualize_frames(extrinsics, images, submap.get_id())

    def export_3d_scene(self, output_path="output.glb"):
        return self.viewer.export(output_path)

    def update_all_submap_vis(self):
        for submap in self.map.get_submaps():
            self.set_submap_point_cloud(submap)
            self.set_submap_poses(submap)

    def update_latest_submap_vis(self):
        submap = self.map.get_latest_submap()
        self.set_submap_point_cloud(submap)
        self.set_submap_poses(submap)

    # -------------------------
    # core SLAM
    # -------------------------
    def add_points(self, pred_dict):
        """
        pred_dict keys:
            images, extrinsic, intrinsic, detected_loops,
            world_points/world_points_conf or depth/depth_conf
        """
        images = pred_dict["images"]                         # (S, 3, H, W)
        extrinsics_cam = pred_dict["extrinsic"]              # (S, 3, 4)
        intrinsics_cam = pred_dict["intrinsic"]              # (S, 3, 3)
        detected_loops = pred_dict.get("detected_loops", [])

        if self.use_point_map:
            world_points = pred_dict["world_points"]         # (S, H, W, 3)
            conf = pred_dict["world_points_conf"]            # (S, H, W)
        else:
            depth_map = pred_dict["depth"]                   # (S, H, W, 1)
            conf = pred_dict["depth_conf"]                   # (S, H, W)
            world_points = unproject_depth_map_to_point_map(depth_map, extrinsics_cam, intrinsics_cam)

        colors = (images.transpose(0, 2, 3, 1) * 255).astype(np.uint8)
        cam_to_world = closed_form_inverse_se3(extrinsics_cam)  # (S, 4, 4)

        assert self.current_working_submap is not None
        new_pcd_num = self.current_working_submap.get_id()

        # --- first edge ---
        if self.first_edge:
            self.first_edge = False

            self.prior_pcd = world_points[-1, ...].reshape(-1, 3)
            self.prior_conf = conf[-1, ...].reshape(-1)

            H_w_submap = np.eye(4)
            self.graph.add_homography(new_pcd_num, H_w_submap)
            self.graph.add_prior_factor(new_pcd_num, H_w_submap, self.graph.anchor_noise)
            H_relative = np.eye(4)

        # --- subsequent edges ---
        else:
            prior_pcd_num = self.map.get_largest_key()
            prior_submap = self.map.get_submap(prior_pcd_num)

            current_pts = world_points[0, ...].reshape(-1, 3)
            curr_conf = conf[0, ...].reshape(-1)

            thr = float(prior_submap.get_conf_threshold())
            good_mask = (self.prior_conf > thr) & (curr_conf > thr)

            min_good = 2000
            n_good0 = int(good_mask.sum())
            fallback_or = 0
            if n_good0 < min_good:
                fallback_or = 1
                print(f"[WARN] good_mask too small ({n_good0} < {min_good}), fallback to OR mask")
                good_mask = (self.prior_conf > thr) | (curr_conf > thr)
            n_good = int(good_mask.sum())

            if self.use_sim3:
                idx = prior_submap.get_last_non_loop_frame_index()
                R_temp = prior_submap.poses[idx][0:3, 0:3]
                t_temp = prior_submap.poses[idx][0:3, 3]

                T_temp = np.eye(4)
                T_temp[0:3, 0:3] = R_temp
                T_temp[0:3, 3] = t_temp
                T_temp = np.linalg.inv(T_temp)

                scale_factor = np.mean(
                    np.linalg.norm((T_temp[0:3, 0:3] @ self.prior_pcd[good_mask].T).T + T_temp[0:3, 3], axis=1) /
                    (np.linalg.norm(current_pts[good_mask], axis=1) + 1e-8)
                )
                print(colored("scale factor", "green"), scale_factor)

                H_relative = np.eye(4)
                H_relative[0:3, 0:3] = R_temp
                H_relative[0:3, 3] = t_temp

                world_points *= scale_factor
                cam_to_world[:, 0:3, 3] *= scale_factor
            else:
                H_relative = ransac_projective(current_pts[good_mask], self.prior_pcd[good_mask])

            # Update prior pcd/conf using last non-loop frame of current submap
            non_lc_frame = self.current_working_submap.get_last_non_loop_frame_index()
            pts_cam0_camn = world_points[non_lc_frame, ...].reshape(-1, 3)
            self.prior_pcd = pts_cam0_camn
            self.prior_conf = conf[non_lc_frame, ...].reshape(-1)

            # Add node to graph
            self.graph.add_homography(new_pcd_num, np.eye(4))

            # Add between factor (prev -> new)
            use_sem_w_odom = (
                self.semantic_weight_mode == "all_edges"
                and self.use_semantic_backend
                and (self.semantic_backend is not None)
                and hasattr(self.graph, "add_between_factor_weighted")
            )

            if use_sem_w_odom:
                sim = 0.0
                w = 1.0
                margin = 0.0
                u = 1.0
                try:
                    p_idx = prior_submap.get_last_non_loop_frame_index()
                    prior_frame = prior_submap.get_frame_at_index(p_idx)
                    curr_frame0 = self.current_working_submap.get_frame_at_index(0)

                    # semantic sim (DINO metric head)
                    sim = float(self.semantic_backend.similarity(curr_frame0, prior_frame))
                    w_sim = self._semantic_sim_to_weight(sim)

                    # uniqueness (retrieval embedding; anti-aliasing for corridor)
                    q_vec = self._get_keyframe_retrieval_vec(self.current_working_submap, 0)
                    margin, u, top1, top2 = self._compute_uniqueness_from_map(
                        q_vec=q_vec,
                        exclude_sid=int(self.current_working_submap.get_id()),
                    )

                    w = float(w_sim) * float(u)
                    w = self._apply_degeneracy_boost(w, n_good)

                    self.graph.add_between_factor_weighted(prior_pcd_num, new_pcd_num, H_relative, w)
                    print(f"[sem_weight][odom] sim={sim:.3f} margin={margin:.3f} u={u:.2f} n_good={n_good} w={w:.2f} {prior_pcd_num}->{new_pcd_num}")
                except Exception as e:
                    print(f"[WARN] odom semantic weight failed -> fallback: {e}")
                    self.graph.add_between_factor(prior_pcd_num, new_pcd_num, H_relative, self.graph.relative_noise)

                self._edge_stats_log(EdgeStat(
                    time=datetime.now().isoformat(timespec="seconds"),
                    edge_type="odom",
                    src=int(prior_pcd_num),
                    dst=int(new_pcd_num),
                    sim=float(sim),
                    w=float(w),
                    n_good=int(n_good),
                    mask_thr=float(thr),
                    mask_fallback_or=int(fallback_or),
                    weight_mode=str(self.semantic_weight_mode),
                    margin=float(margin),
                    u=float(u),
                ))
            else:
                self.graph.add_between_factor(prior_pcd_num, new_pcd_num, H_relative, self.graph.relative_noise)

            print("added between factor", prior_pcd_num, new_pcd_num)
            H_w_submap = prior_submap.get_reference_homography() @ H_relative

        # --- create/update current submap ---
        self.current_working_submap.set_reference_homography(H_w_submap)
        self.current_working_submap.add_all_poses(cam_to_world)
        self.current_working_submap.add_all_points(
            world_points,
            colors,
            conf,
            self.init_conf_threshold,
            intrinsics_cam,
        )
        self.current_working_submap.set_conf_masks(conf)

        # --- loop closures (if any) ---
        if detected_loops is not None and len(detected_loops) > 0:
            for index, loop in enumerate(detected_loops):
                if hasattr(loop, "query_submap_id"):
                    assert loop.query_submap_id == self.current_working_submap.get_id()

                loop_index = self.current_working_submap.get_last_non_loop_frame_index() + index + 1

                if self.use_sim3:
                    pose_world_detected = self.map.get_submap(loop.detected_submap_id).get_pose_subframe(loop.detected_submap_frame)
                    pose_world_query = self.current_working_submap.get_pose_subframe(loop_index)
                    pose_world_detected = gtsam.Pose3(pose_world_detected)
                    pose_world_query = gtsam.Pose3(pose_world_query)
                    H_relative_lc = pose_world_detected.between(pose_world_query).matrix()
                else:
                    points_world_detected = self.map.get_submap(loop.detected_submap_id).get_frame_pointcloud(loop.detected_submap_frame).reshape(-1, 3)
                    points_world_query = self.current_working_submap.get_frame_pointcloud(loop_index).reshape(-1, 3)
                    H_relative_lc = ransac_projective(points_world_query, points_world_detected)

                # 关键：避免重复加边（weighted 就不要再 add normal）
                use_sem_w_loop = (
                    self.semantic_weight_mode in ("loop_only", "all_edges")
                    and self.use_semantic_backend
                    and (self.semantic_backend is not None)
                    and hasattr(self.graph, "add_between_factor_weighted")
                )

                if use_sem_w_loop:
                    sim = 0.0
                    w = 1.0
                    loop_margin = float(getattr(loop, "semantic_margin", 0.0))
                    loop_u = float(getattr(loop, "semantic_u", 1.0))
                    try:
                        sim_cached = getattr(loop, "semantic_sim", None)
                        if sim_cached is None:
                            q_frame = self.current_working_submap.get_frame_at_index(loop_index)
                            d_submap = self.map.get_submap(loop.detected_submap_id)
                            d_frame = d_submap.get_frame_at_index(loop.detected_submap_frame)
                            sim = float(self.semantic_backend.similarity(q_frame, d_frame))
                        else:
                            sim = float(sim_cached)

                        # Down-weight ambiguous loops using the semantic margin-derived
                        # uniqueness factor (u) computed during loop gating.
                        loop_margin = float(getattr(loop, "semantic_margin", 0.0))
                        loop_u = float(getattr(loop, "semantic_u", 1.0))

                        w = float(self._semantic_sim_to_weight(sim)) * float(loop_u)
                        self.graph.add_between_factor_weighted(loop.detected_submap_id, loop.query_submap_id, H_relative_lc, w)
                        print(
                            f"[sem_weight][loop] sim={sim:.3f} margin={loop_margin:.3f} u={loop_u:.2f} w={w:.2f} "
                            f"{loop.detected_submap_id}->{loop.query_submap_id}"
                        )
                    except Exception as e:
                        print(f"[WARN] loop semantic weight failed -> fallback: {e}")
                        self.graph.add_between_factor(loop.detected_submap_id, loop.query_submap_id, H_relative_lc, self.graph.relative_noise)

                    self._edge_stats_log(EdgeStat(
                        time=datetime.now().isoformat(timespec="seconds"),
                        edge_type="loop",
                        src=int(loop.detected_submap_id),
                        dst=int(loop.query_submap_id),
                        sim=float(sim),
                        w=float(w),
                        n_good=0,
                        mask_thr=0.0,
                        mask_fallback_or=0,
                        weight_mode=str(self.semantic_weight_mode),
                        margin=float(getattr(loop, "semantic_margin", 0.0)),
                        u=float(getattr(loop, "semantic_u", 1.0)),
                    ))
                else:
                    self.graph.add_between_factor(loop.detected_submap_id, loop.query_submap_id, H_relative_lc, self.graph.relative_noise)

                self.graph.increment_loop_closure()
                print("added loop closure factor", loop.detected_submap_id, loop.query_submap_id)

                print(
                    "homography between nodes estimated to be",
                    np.linalg.inv(self.map.get_submap(loop.detected_submap_id).get_reference_homography()) @ H_w_submap,
                )

        # finally, add submap into map
        self.map.add_submap(self.current_working_submap)

    def sample_pixel_coordinates(self, H, W, n):
        y_coords = torch.randint(0, H, (n,), dtype=torch.float32)
        x_coords = torch.randint(0, W, (n,), dtype=torch.float32)
        pixel_coords = torch.stack((y_coords, x_coords), dim=1)
        return pixel_coords

    # -------------------------
    # semantic gating
    # -------------------------
    def _filter_loops_by_semantic(self, new_submap, detected_loops):
        """
        Filter detected_loops using semantic similarity between query frame and retrieved frame.
        会把 sim 缓存到 loop.semantic_sim，后续 loop reweighting 直接用。
        """
        if self.semantic_gate_mode not in ("filter", "both"):
            return detected_loops
        if (not self.use_semantic_backend) or (self.semantic_backend is None):
            return detected_loops
        if detected_loops is None or len(detected_loops) == 0:
            return detected_loops

        kept = []
        thr = float(self.semantic_min_sim)

        for loop in detected_loops:
            try:
                q_idx = getattr(loop, "query_submap_frame", None)
                if q_idx is None:
                    q_idx = getattr(loop, "query_frame", None)
                if q_idx is None:
                    q_idx = len(new_submap.get_frame_ids()) - 1

                d_sid = getattr(loop, "detected_submap_id", None)
                d_fid = getattr(loop, "detected_submap_frame", None)
                if (d_sid is None) or (d_fid is None):
                    kept.append(loop)
                    continue

                q_frame = new_submap.get_frame_at_index(int(q_idx))
                d_frame = self.map.get_submap(d_sid).get_frame_at_index(int(d_fid))

                sim = float(self.semantic_backend.similarity(q_frame, d_frame))
                setattr(loop, "semantic_sim", sim)

                if sim >= thr:
                    kept.append(loop)
            except Exception:
                kept.append(loop)

        return kept

    def run_predictions(self, image_names, model, max_loops):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        images = load_and_preprocess_images(image_names).to(device)
        print(f"Preprocessed images shape: {images.shape}")

        dtype = torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8) else torch.float16

        new_pcd_num = self.map.get_largest_key() + 1
        new_submap = Submap(new_pcd_num)
        new_submap.add_all_frames(images)
        new_submap.set_frame_ids(image_names)

        # retrieval vectors for loop closure
        self._set_submap_attr(new_submap, 'retrieval_vectors', self.image_retrieval.get_all_submap_embeddings(new_submap))

        # IMPORTANT: set before loop closure to keep frame indexing consistent
        new_submap.set_last_non_loop_frame_index(images.shape[0] - 1)

        detected_loops = self.image_retrieval.find_loop_closures(self.map, new_submap, max_loop_closures=max_loops)

        # semantic filter on loop candidates
        if self.semantic_gate_mode in ("filter", "both") and len(detected_loops) > 0:
            before = len(detected_loops)
            detected_loops = self._filter_loops_by_semantic(new_submap, detected_loops)
            after = len(detected_loops)
            print(colored("semantic_loop_filter", "cyan"), f"{before}->{after}")

        if len(detected_loops) > 0:
            print(colored("detected_loops", "yellow"), detected_loops)

        retrieved_frames = self.map.get_frames_from_loops(detected_loops)

        # semantic gate on retrieved frames (query frame vs retrieved frame)
        if self.semantic_gate_mode in ("retrieved", "both") and len(detected_loops) > 0 and len(retrieved_frames) > 0:
            try:
                q_ids = new_submap.get_frame_ids()
                q_idx = len(q_ids) - 1
                q_frame = new_submap.get_frame_at_index(q_idx)

                sims = [float(self.semantic_backend.similarity(q_frame, rf)) for rf in retrieved_frames]
                thr = float(self.semantic_min_sim)

                # cache sim to loop object if possible
                for i, sim in enumerate(sims):
                    try:
                        setattr(detected_loops[i], "semantic_sim", float(sim))
                    except Exception:
                        pass

                # 1) threshold filter
                cand = [(i, sims[i]) for i in range(len(sims)) if sims[i] >= thr]
                before = len(detected_loops)
                if len(cand) == 0:
                    detected_loops, retrieved_frames = [], []
                    after = 0
                    print(f"[semantic_loop_gate] min_sim={thr:.3f} {before}->{after} (no candidate) sims={[round(x,3) for x in sims]}")
                else:
                    # 2) rank + compute ambiguity margin (best-second best)
                    cand.sort(key=lambda x: x[1], reverse=True)
                    best_i, best_sim = cand[0]
                    second_sim = cand[1][1] if len(cand) > 1 else -1.0
                    margin = float(best_sim - second_sim) if len(cand) > 1 else 1.0

                    # Map margin -> u in (u_min, 1]; small margin => ambiguous => down-weight.
                    u = float(self._compute_u(margin, scale=self.semantic_u_scale, u_min=self.semantic_u_min))

                    topk = max(1, int(self.semantic_loop_topk))
                    keep = [i for (i, _) in cand[:topk]]

                    # Attach margin/u to kept loop objects for downstream weighting & logging.
                    for i in keep:
                        try:
                            setattr(detected_loops[i], "semantic_margin", float(margin))
                            setattr(detected_loops[i], "semantic_u", float(u))
                        except Exception:
                            pass

                    detected_loops = [detected_loops[i] for i in keep]
                    retrieved_frames = [retrieved_frames[i] for i in keep]
                    after = len(detected_loops)
                    print(
                        f"[semantic_loop_gate] min_sim={thr:.3f} topk={topk} {before}->{after} "
                        f"best={best_sim:.3f} second={second_sim:.3f} margin={margin:.3f} u={u:.3f} "
                        f"keep={keep} sims_head={[round(x,3) for x in sims[:10]]}"
                    )
            except Exception as e:
                print(f"[WARN] semantic_loop_gate failed (keep loops): {e}")

        num_loop_frames = len(retrieved_frames)

        if num_loop_frames > 0:
            image_tensor = torch.stack(retrieved_frames)      # (n, 3, H, W)
            images = torch.cat([images, image_tensor], dim=0) # (S+n, 3, H, W)
            new_submap.add_all_frames(images)

        self.current_working_submap = new_submap

        with torch.no_grad():
            with torch.cuda.amp.autocast(dtype=dtype):
                predictions = model(images)

        extrinsic, intrinsic = pose_encoding_to_extri_intri(predictions["pose_enc"], images.shape[-2:])
        predictions["extrinsic"] = extrinsic
        predictions["intrinsic"] = intrinsic
        predictions["detected_loops"] = detected_loops

        for key in list(predictions.keys()):
            if isinstance(predictions[key], torch.Tensor):
                predictions[key] = predictions[key].cpu().numpy().squeeze(0)

        return predictions


```

### vggt_slam/submap.py

```python
import re
import os
import cv2
import torch
import numpy as np
import open3d as o3d


class Submap:
    def __init__(self, submap_id):
        self.submap_id = submap_id

        # Geometry / pose
        self.H_world_map = None            # (4,4)
        self.R_world_map = None
        self.poses = None                  # (N,4,4)

        # Frame-level data
        self.frames = None                 # (N,H,W,3)
        self.frame_ids = None
        self.last_non_loop_frame_index = None

        # Camera / intrinsics
        self.vggt_intrinscs = None          # (3,4) or (3,3)

        # Semantic / retrieval
        self.retrieval_vectors = None

        # Dense geometry
        self.pointclouds = None             # (N,H,W,3)
        self.colors = None                  # (N,H,W,3)
        self.conf = None                    # (N,H,W)
        self.conf_masks = None              # (N,H,W)
        self.conf_threshold = None

        # Cached
        self.voxelized_points = None

    # ------------------------------------------------------------------
    # Basic setters
    # ------------------------------------------------------------------

    def add_all_poses(self, poses):
        self.poses = poses

    def add_all_frames(self, frames):
        self.frames = frames

    def set_all_retrieval_vectors(self, retrieval_vectors):
    # 兼容 solver.py 里调用的旧/新接口命名
        self.add_all_retrieval_vectors(retrieval_vectors)

    def get_all_retrieval_vectors(self):
        return self.retrieval_vectors


    def add_all_retrieval_vectors(self, retrieval_vectors):
        self.retrieval_vectors = retrieval_vectors

    def add_all_points(self, points, colors, conf,
                       conf_threshold_percentile,
                       intrinsics):
        """
        points: (N,H,W,3)
        colors: (N,H,W,3)
        conf:   (N,H,W)
        """
        self.pointclouds = points
        self.colors = colors
        self.conf = conf
        self.conf_threshold = np.percentile(conf, conf_threshold_percentile)
        self.vggt_intrinscs = intrinsics

    # ------------------------------------------------------------------
    # ID / meta
    # ------------------------------------------------------------------

    def get_id(self):
        return self.submap_id

    def set_frame_ids(self, file_paths):
        """
        Extract numeric frame ids from filenames.
        Example: frame_00321.jpg -> 321
        """
        frame_ids = []
        for path in file_paths:
            name = os.path.basename(path)
            match = re.search(r"\d+(?:\.\d+)?", name)
            if match:
                frame_ids.append(float(match.group()))
            else:
                raise ValueError(f"No number found in filename: {name}")
        self.frame_ids = frame_ids

    def get_frame_ids(self):
        return self.frame_ids

    def set_last_non_loop_frame_index(self, idx):
        self.last_non_loop_frame_index = idx

    def get_last_non_loop_frame_index(self):
        return self.last_non_loop_frame_index

    # ------------------------------------------------------------------
    # Reference transform
    # ------------------------------------------------------------------

    def set_reference_homography(self, H_world_map):
        self.H_world_map = H_world_map

    def get_reference_homography(self):
        return self.H_world_map

    # ------------------------------------------------------------------
    # Frame access
    # ------------------------------------------------------------------

    def get_frame_at_index(self, index):
        return self.frames[index]

    def get_all_frames(self):
        return self.frames

    # ------------------------------------------------------------------
    # Pose utilities
    # ------------------------------------------------------------------

    def get_pose_subframe(self, pose_index):
        return np.linalg.inv(self.poses[pose_index])

    def get_all_poses_world(self, ignore_loop_closure_frames=False):
        """
        Recover camera poses in world frame using projection decomposition.
        """
        proj_mats = (
            self.vggt_intrinscs
            @ np.linalg.inv(self.poses)[:, 0:3, :]
            @ np.linalg.inv(self.H_world_map)
        )

        poses_world = []
        for i, P in enumerate(proj_mats):
            K, R, t = cv2.decomposeProjectionMatrix(P)[0:3]
            t = t / t[3, 0]

            T = np.eye(4)
            T[0:3, 0:3] = np.linalg.inv(R)
            T[0:3, 3] = t[0:3, 0]

            poses_world.append(T)

            if ignore_loop_closure_frames and i == self.last_non_loop_frame_index:
                break

        return np.stack(poses_world, axis=0)

    # ------------------------------------------------------------------
    # Confidence filtering
    # ------------------------------------------------------------------

    def set_conf_masks(self, conf_masks):
        self.conf_masks = conf_masks

    def get_conf_threshold(self):
        return self.conf_threshold

    def filter_data_by_confidence(self, data, stride=1):
        """
        Generic confidence filter for points / colors.
        """
        if stride == 1:
            mask = self.conf >= self.conf_threshold
            return data[mask]
        else:
            conf_sub = self.conf[:, ::stride, ::stride]
            data_sub = data[:, ::stride, ::stride, :]
            mask = conf_sub >= self.conf_threshold
            return data_sub[mask]

    # ------------------------------------------------------------------
    # Point cloud access
    # ------------------------------------------------------------------

    def get_frame_pointcloud(self, pose_index):
        return self.pointclouds[pose_index]

    def get_points_colors(self, stride=1):
        colors = self.filter_data_by_confidence(self.colors, stride)
        return colors.reshape(-1, 3)

    def get_points_in_world_frame(self, stride=1):
        points = self.filter_data_by_confidence(self.pointclouds, stride)

        pts = points.reshape(-1, 3)
        pts_h = np.hstack([pts, np.ones((pts.shape[0], 1))])

        pts_w = (self.H_world_map @ pts_h.T).T
        return pts_w[:, :3] / pts_w[:, 3:]

    def get_points_list_in_world_frame(self, ignore_loop_closure_frames=False):
        """
        Return per-frame world points + frame ids + confidence masks
        """
        point_list = []
        frame_id_list = []
        frame_conf_mask = []

        for i, pts in enumerate(self.pointclouds):
            pts_flat = pts.reshape(-1, 3)
            pts_h = np.hstack([pts_flat, np.ones((pts_flat.shape[0], 1))])
            pts_w = (self.H_world_map @ pts_h.T).T
            pts_w = (pts_w[:, :3] / pts_w[:, 3:]).reshape(pts.shape)

            point_list.append(pts_w)
            frame_id_list.append(self.frame_ids[i])
            frame_conf_mask.append(self.conf_masks[i] >= self.conf_threshold)

            if ignore_loop_closure_frames and i == self.last_non_loop_frame_index:
                break

        return point_list, frame_id_list, frame_conf_mask

    # ------------------------------------------------------------------
    # Voxelized point cloud (Open3D)
    # ------------------------------------------------------------------

    def get_voxel_points_in_world_frame(
        self,
        voxel_size,
        nb_points=8,
        factor_for_outlier_rejection=2.0,
    ):
        if self.voxelized_points is None:
            if voxel_size <= 0:
                raise RuntimeError("voxel_size must be > 0")

            pts = self.filter_data_by_confidence(self.pointclouds)
            cols = self.filter_data_by_confidence(self.colors)

            pts = pts.reshape(-1, 3)
            cols = (cols.reshape(-1, 3)) / 255.0

            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(pts)
            pcd.colors = o3d.utility.Vector3dVector(cols)

            pcd = pcd.voxel_down_sample(voxel_size)

            if nb_points > 0:
                pcd, _ = pcd.remove_radius_outlier(
                    nb_points=nb_points,
                    radius=voxel_size * factor_for_outlier_rejection,
                )

            self.voxelized_points = pcd

        pts = np.asarray(self.voxelized_points.points)
        pts_h = np.hstack([pts, np.ones((pts.shape[0], 1))])
        pts_w = (self.H_world_map @ pts_h.T).T

        pcd_w = o3d.geometry.PointCloud()
        pcd_w.points = o3d.utility.Vector3dVector(
            pts_w[:, :3] / pts_w[:, 3:]
        )
        pcd_w.colors = self.voxelized_points.colors

        return pcd_w

```

