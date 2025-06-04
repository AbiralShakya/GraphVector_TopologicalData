JARVIS database used widely for generative models in traditional "AI for materials" resarch groups

Outside of spin orbit coupling data (loosely related but not direct data / ground truth) there's no data on topological 

although topological quantum chemistry database exists with .poscar files of atomic structures, no direct path to retrieve properties from jarvis and directly analyze .poscar files easily of graph structure / lattice structure 

this project aims to combine JARVIS & topological data with a unified graph-vector database structure for further work in generative models for topological materials 


EBR from TQC --> symmetry and interconnections in k space + poscar based lattice structure combined with jarvis data 
    change to vector database (jarvis + TQC regular)
    accompanby with corresponding graph based structure as stated above 

Revised insight: 
    lower computational power & energy load w/ engi

    prev generative modelling for TI have all been constrained to TQC, we change that, a lot
        more data, physics informed, crystal structure aware, band structure & topology aware (constructed as graph in db so save computational time in training / inference)


https://www.nature.com/articles/s41535-025-00731-0 -- vanilla
https://www.nature.com/articles/s41578-021-00380-2?fromPaywallRec=false#Sec15 -- review of physics & db
https://www.nature.com/articles/s41524-025-01592-8?fromPaywallRec=false#data-availability -- constraint / target feature
https://www.nature.com/articles/s41524-023-00987-9?fromPaywallRec=false -- Phy Inform dl

can use 3d crystal graphs for 2d generative modelling later (isolate monolayers, train multi task classifier)

https://www.cryst.ehu.es/#structuretop is my goat, downloaded local files


the hacky method: just change ICSD id url lol
    i.e. https://www.topologicalquantumchemistry.fr/#/detail/3
        & https://www.topologicalquantumchemistry.com/data/nonalloycompounds/SG-165/Mat-3/POSCAR
        
    k-paths 

https://static-content.springer.com/esm/art%3A10.1038%2Fnature23268/MediaObjects/41586_2017_BFnature23268_MOESM1_ESM.pdf -- download local files for space group BANDREP


TODO: 
    by 06.05.2025 EOD
        ~ finish all DB creation   
            - BANDREP and magnetic BANDREP
            - topological indices for single and double irrep 
            - create .db files for the 4 combos
            - relate to local files
            - pull from JARVIS
            - create graph structure of data
            - create full DB
        
        ~ look at prev multi task code and start building mag + topo multi task (worked on before) 
            - read physics + ml papers on 2d materials classification, classification of electronic properties, magnetic properties when applied currents to 2d or 3d lattice
        
        - important: understand an SOTA ml classifier regarding TI, look at runtime, compare to DFT methods, idea for creating my own algo to do such classification and how to compare training workload