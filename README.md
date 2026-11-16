# Image-Colorization

## Repository Structure

```
Image-Colorization/
├── baselines/
│   ├── zhang2016/          
│   └── sd_lora/            
├── color_image/            
│   ├── assets/
│   │   └── pts_in_hull.npy  
│   ├── baseline.py
│   ├── common.py
│   ├── config.py
│   ├── infer.py
│   ├── model.py
│   └── train.py
├── scripts/
│   ├── base.py
│   └── download_coco.py     
├── data/                   
│   └── coco/
│       ├── images/          
│       ├── splits/{train,val,test}.txt   
│       ├── captions.json    
│       └── annotations_trainval2017.zip
├── pyproject.toml 
├── setup.py
└── .gitmodules
```

---