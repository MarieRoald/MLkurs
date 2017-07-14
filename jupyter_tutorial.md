
# Installere og kjøre Python

## Installasjon av Anaconda

I dette kurset skal vi bruke Python. For å gjøre installasjonen så smertefri
som mulig, skal vi bruke en distribusjon som heter Anaconda. Anaconda
inneholder det vi trenger av pakker, så vi behøver ikke installere noe mer.

Du laster ned Anaconda [herfra](https://www.continuum.io/downloads). Vi skal
bruke Python 3.6.  Installasjonen skjer som installasjonen av et hvilket som
helst annet program.

## Kjøring av Jupyter notebook

Programmet vi skal bruke til å programmere i, heter Jupyter. Det er et program
som kjører i nettleseren din, og er utformet for å kombinere kode og tekst, og
også for å gjøre det lett å visualisere ting.

Jupyter følger med Anaconda, men må startes manuelt. Dette kan gjøres på to
forskjellige måter:

### Alternativ 1

Du skal nå har et program som heter "Jupyter Notebook". Dette finner du
forhåpentligvis ved å søke på maskinen din. Når Jupyter starter, får du opp ett
terminalvindu.  Deretter åpnes en ny fane i nettleseren din. Du skal nå være i
hjemmeområdet ditt, og du kan navigere deg til der du har lagret
kursmateriellet.

### Alternativ 2

Dette er en litt mer avansert måte å gjøre det på. Start Anaconda (f.eks. ved å
søke etter det). Deretter kan du navigere til mappen der du har lagret filene
ved å bruke `cd` (change directory). `cd mappenavn` tar deg til `mappenavn`
hvis den mappen finnes der du er.  `cd ..` tar deg ett nivå opp.

Hvis du har mappen du vil navigere til åpen i et Windows Explorer-vindu (altså
filutforskeren, ikke nettleseren), kan du trykke på mappesymbolet ved linjen
som viser hvor du er.  Da vil innholdet i linjen endre seg til å vise
plasseringen din i ren tekst. Du kan kopiere denne, og lime inn etter `cd` i
terminalen. Da vil den ta deg til denne plasseringen.

Du kan se innholdet av mappen du er i med `dir` på Windows og `ls` på
Mac/Linux.  Hvis du vil se hvilken mappe du er i, skriver du `cd` på Windows og
`pwd` på Mac/Linux.

Når du har kommet deg inn i riktig mappe, skriver du `jupyter notebook`. Det
samme vil skje som for alternativ 1, bare at du nå starter i den mappen du
startet Jupyter Notebook fra.

## Avslutte Jupyter Notebook

Du avslutter Jupyter notebook ved å trykke Ctrl-C i terminalvinduet der det
kjører. Du vil da få spørsmål om du vil avslutte. Trykk `y` for 'yes', og trykk
`Enter`.
