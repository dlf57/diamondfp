# diamondfp
**Comparing baseball players like molecules!** 

**Statistical fingerprints for baseball players — find their closest matches.**  

`diamondfp` is a Python library for generating **statistical fingerprints** of baseball players, enabling similarity searches much like molecular similarity in chemistry. By encoding player performance into binary feature vectors, you can compare players across eras and styles using metrics like the **Tanimoto coefficient/Jaccard Score**.  

Currently, `diamondfp` supports **binary fingerprint generation** for batting stats, with pitching and fielding fingerprints in development.  

## Features  
- Create binary statistical fingerprints from player performance data  
- Compare players using similarity metrics (Tanimoto coefficient/Jaccard Score)  
- Easily extendable to batting, pitching, and fielding statistics  
- Works with historical and modern baseball data 

## Usage

```python
import pandas as pd
from diamondfp.binary_fp import gen_feat_quants, binary_fp
from diamondfp.scoring import jaccard

df = pd.read_csv('career-stats.csv')
basic_features = [
    "2B",
    "3B",
    "RBI",
    "SB",
    "K%",
    "BB%"
]
advanced_features = [
    "H",
    "HR",
    "AVG",
    "OBP",
    "SLG",
    "OPS",
]
basic_quants = [0.5, 0.75]
advanced_quants = [0.5, 0.75, 0.9, 0.95]


feat_quants = gen_feat_quants(df, basic_features, advanced_features, basic_quants, advanced_quants)
df['diamondFP'] = df.apply(lambda x: binary_fp(x, feat_quants), axis=1)

babe_ruth = df[df['Name'] == "Babe Ruth"]["diamondFP"].to_list()[0]
shohei_ohtani = df[df['Name'] == "Shohei Ohtani"]["diamondFP"].to_list()[0]
sim_score = jaccard(babe_ruth, shohei_ohtani)
print(sim_score) # 0.86
```

## Example Data Source

Example notebooks use the Lahman Baseball Database,
licensed under Creative Commons BY-SA 3.0.

### Roadmap

- [X] Binary vector fingerprints
- [] Binned fingerprints
- [] Clustering and visualization tools
- [] PyPI release

## License
This project is licensed under the MIT License. See the [LICENSE](https://github.com/dlf57/diamondfp/blob/main/LICENSE) file for details.