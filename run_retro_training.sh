#!/bin/bash

### Stable-retro
envs=(
    ### Platformer games
    # "SonicTheHedgehog2-Genesis-v0"
    # "SonicTheHedgehog3-Genesis-v0"
    # "SonicAndKnuckles3-Genesis-v0"
    # "SuperMarioBros3-Nes-v0"
#     "Ristar-Genesis-v0"
#     "RocketKnightAdventures-Genesis-v0"
#     "CastleOfIllusion-Genesis-v0"
#     "QuackShot-Genesis-v0"
#     "Vectorman2-Genesis-v0"
#     "KidChameleon-Genesis-v0"
#     "CoolSpot-Genesis-v0"
#     "GreendogTheBeachedSurferDude-Genesis-v0"
#     "KirbysAdventure-Nes-v0"
#     "MegaMan2-Nes-v0"
#     "AdventureIsland3-Nes-v0"
#     "FelixTheCat-Nes-v0"
#     "LittleMermaid-Nes-v0"
#     "BuckyOHare-Nes-v0"
#     "KidIcarus-Nes-v0"
#     "Shatterhand-Nes-v0"
#     "RockinKats-Nes-v0"
#     "ViceProjectDoom-Nes-v0"
#     "BubsyII-Snes-v0"
#     "ActRaiser2-Snes-v0"
#     "Plok-Snes-v0"

    ### Sport games
    # "SuperHangOn-Genesis-v0"
    # "NHL94-Genesis-v0"
    "F1-Genesis-v0"
    "EuropeanClubSoccer-Genesis-v0"

    ### Arcade shooters
    # "BioHazardBattle-Genesis-v0"
    # "MUSHA-Genesis-v0"
    # "Truxton-Genesis-v0"
#     "GrindStormer-Genesis-v0"
#     "Hellfire-Genesis-v0"
#     "Gaiares-Genesis-v0"
#     "ElementalMaster-Genesis-v0"
#     "ZeroWing-Genesis-v0"
#     "Viewpoint-Genesis-v0"
#     "SteelEmpire-Genesis-v0"
#     "GradiusII-Nes-v0"
#     "LifeForce-Nes-v0"
#     "Zanac-Nes-v0"
#     "GunNac-Nes-v0"
#     "TwinBee-Nes-v0"
#     "Parodius-Nes-v0"
#     "TerraCresta-Nes-v0"
#     "BuraiFighter-Nes-v0"
#     "DragonSpiritTheNewLegend-Nes-v0"
#     "XeviousTheAvenger-Nes-v0"
#     "Jackal-Nes-v0"
#     "HeavyBarrel-Nes-v0"
#     "GuerrillaWar-Nes-v0"
#     "POWPrisonersOfWar-Nes-v0"
#     "SuperC-Nes-v0"
    "AeroFighters-Snes-v0"

    ### Action games
    # "StreetsOfRage3-Genesis-v0"
    # "GoldenAxeIII-Genesis-v0"
    # "TeenageMutantNinjaTurtlesTheHyperstoneHeist-Genesis-v0"
#    "DoubleDragonIITheRevenge-Nes-v0"
#    "TeenageMutantNinjaTurtlesIIITheManhattanProject-Nes-v0"
#    "FinalFight3-Snes-v0"

    ### Puzzle / Classic games - HOTOVO
     # "MsPacMan-Genesis-v0"
#     "PacMania-Genesis-v0"
#     "BalloonFight-Nes-v0"
     "DonkeyKong-Nes-v0"
#     "BubbleBobble-Nes-v0"
#     "SnowBrothers-Nes-v0"
#     "Arkanoid-Nes-v0"
#     "Popeye-Nes-v0"
#     "BoulderDash-GameBoy-v0"
#     "GradiusTheInterstellarAssault-GameBoy-v0"
#     "BlockKuzushiGB-GameBoy-v0"
#     "Cameltry-Snes-v0"
#     "PacInTime-Snes-v0"
)

# Loop through each environment and run the trainer
for env in "${envs[@]}"; do
    echo "--------------------------------------------------"
    echo "Starting training for: $env"
    echo "--------------------------------------------------"

    python3 ./sb3_agents/trainer.py --emulator retro --env "$env"

    echo "Finished training for: $env"
done