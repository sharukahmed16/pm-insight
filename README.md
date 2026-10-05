# PM-Insight

Story point estimation for agile work items using machine learning.
MITM 421 Project, IIT, University of Dhaka.

## Setup

python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt

## Data

Deep-SE story point dataset, Choetkiertikul et al., IEEE TSE 2019.
Not committed to this repository. To obtain it:
git clone --depth 1 https://github.com/morakotch/datasets.git
Copy the 16 CSV files from storypoint/IEEE TSE2018/dataset/ into data/raw/
