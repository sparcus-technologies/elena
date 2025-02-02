# ELENA: Epigenetic Learning through Evolved Neural Adaptation
Code accompaniment for the research work on ELENA: Epigenetic Learning through Evolved Neural Adaptation

Preprint can be found here: https://arxiv.org/abs/2501.05735 <br>

# Abstract <br>
Despite the success of metaheuristic algorithms in solving complex network optimization problems, they often struggle with adaptation, especially in dynamic or high-dimensional search spaces. Traditional approaches can become stuck in local optima, leading to inefficient exploration and suboptimal solutions. Most of the widely accepted advanced algorithms do well either on highly complex or smaller search spaces due to the lack of adaptation. To address these limitations, we present ELENA (Epigenetic Learning through Evolved Neural Adaptation), a new evolutionary framework that incorporates epigenetic mechanisms to enhance the adaptability of the core evolutionary approach. ELENA leverages compressed representation of learning parameters improved dynamically through epigenetic tags that serve as adaptive memory. Three epigenetic tags (mutation resistance, crossover affinity, and stability score) assist with guiding solution space search, facilitating a more intelligent hypothesis landscape exploration. To assess the framework’s performance, we conduct experiments on three critical network optimization problems: the Traveling Salesman Problem (TSP), the Vehicle Routing Problem (VRP), and the Maximum Clique Problem (MCP). Experiments indicate that ELENA achieves competitive results, often surpassing state-of-the-art methods on network optimization tasks.

# Contributors <br>
*Kriuk Boris, Hong Kong University of Science and Technology* <br>
*Sulamanidze Keti, IE University* <br>
*Kriuk Fedor, University of Technology Sydney* <br>

# Data Availability <br>
1. Augerat-1995 set A: http://www.vrp-rep.org/datasets/item/2014-0000.html. <br>
1. Augerat-1995 set P: http://www.vrp-rep.org/datasets/item/2014-0009.html. <br>

# Repository Structure <br>
1. ELENA_TSP.ipynb: Code for ELENA implementation on TSP with performance comparison. <br>
2. Others_TSP.ipynb: Code for common algorithms implementation on TSP with performance comparison. <br>
3. ELENA-300-0.5_VRP_Augerat_A.ipynb: ELENA-300-0.5 implementation on VRP on Augerat-1995 set A. <br>
4. ELENA-300-0.5_VRP_Augerat_P.ipynb: ELENA-300-0.5 implementation on VRP on Augerat-1995 set P. <br>
5. Others_VRP_Augerat_A.ipynb: Code for common algorithms implementation on VRP on Augerat-1995 set A. <br>
6. Others_VRP_Augerat_P.ipynb: Code for common algorithms implementation on VRP on Augerat-1995 set P. <br>
7. ELENA_MCP.ipynb: Code for ELENA implementation on MCP with performance comparison. <br>
8. ELENA_Streamlit_Visualization: Code for interactive ELENA implementation as a Streamlit app. <br>

README.md: This file, containing an overview of the repository. <br>
LICENSE: Apache 2.0 License. <br>
