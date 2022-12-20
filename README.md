# PaperEdge
<a href="https://huggingface.co/spaces/SWHL/PaperEdgeDemo"><img src="https://img.shields.io/badge/%F0%9F%A4%97-Open%20in%20Spaces-blue"></a><br/>
The code and the DIW dataset for "Learning From Documents in the Wild to Improve Document Unwarping" (SIGGRAPH 2022)

[[paper](https://drive.google.com/file/d/1z_8YaCc3aGWTHqFP55vgpSaEBz_oaQQe/view?usp=sharing)]
[[supplementary material](https://drive.google.com/file/d/1szMa0D90E9caKVwonnZFduTpla52sKEz/view?usp=sharing)]
![image](https://user-images.githubusercontent.com/12742725/177686793-77c6652e-f86a-45ea-829f-78306f2d5021.png)


## Documents In the Wild (DIW) dataset (2.13GB)
[link](https://drive.google.com/file/d/1qAmLurt6bK0ro8PnRz6rBgVs1rfrsdKi/view?usp=sharing)

## Pretrained models (139.7MB each)

[Enet](https://drive.google.com/file/d/1OVHETBHQ5u-1tnci3qd7OcAjas4v1xnl/view?usp=sharing)

[Tnet](https://drive.google.com/file/d/1gEp4ecmdvKds2nzk9CaZb_pLvhRoyAsv/view?usp=sharing)

## DocUNet benchmark results
[docunet_benchmark_paperedge.zip](https://drive.google.com/file/d/1QM3Y5Ty96ydVCQPNqR0_bnMG9oqIQkGm/view?usp=sharing)

The last row of `adres.txt` is the evaluation results.
The values in the last 3 columns are `AD`, `MS-SSIM`, and `LD`.

## Infer one image.
1. Download the pretrained model to the `models` directory.
2. Run the `demo.py` by the following code:
    ```shell
    $ python demo.py --Enet_ckpt 'models/G_w_checkpoint_13820.pt' \
                     --Tnet_ckpt 'models/L_w_checkpoint_27640.pt' \
                     --img_path 'images/1.jpg' \
                     --out_dir 'output'
    ```
  3. The final result:
  ![compare](https://user-images.githubusercontent.com/28639377/196933170-81c7e3d8-3661-429b-ae17-efae33366545.png)
