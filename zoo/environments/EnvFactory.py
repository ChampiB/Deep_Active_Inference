import gym
from stable_baselines3.common.monitor import Monitor
from hydra.utils import instantiate
from zoo.environments.wrappers.PytorchWrapper import PytorchWrapper
from zoo.environments.wrappers.ResizeWrapper import ResizeWrapper


def _get_custom_env(config):
    """
    Create the custom environment according to the configuration
    :param config: the hydra configuration
    :return: the created environment, or None if the environment is not a custom environment
    """

    # Get the environment name.
    env_name = config.environment.name

    # Create the environment, if its name is in the list of custom environments, otherwise return None
    for env in ["d_sprites", "maze"]:
        if env_name.startswith(env):
            return instantiate(config.environment)
    return None


def make(config):
    """
    Create the environment according to the configuration
    :param config: the hydra configuration
    :return: the created environment
    """

    # List of Atari games that could be added to the framework.
    # atari_games = [
    #     'ALE/Videocube-v5', 'ALE/DonkeyKong-v5', 'ALE/Pitfall2-v5', 'ALE/Koolaid-v5', 'ALE/NameThisGame-v5',
    #     'ALE/Enduro-v5', 'ALE/Et-v5', 'ALE/MiniatureGolf-v5', 'ALE/KingKong-v5', 'ALE/RoadRunner-v5',
    #     'ALE/TicTacToe3D-v5', 'ALE/Asteroids-v5', 'ALE/SpaceInvaders-v5', 'ALE/ElevatorAction-v5', 'ALE/Galaxian-v5',
    #     'ALE/AirRaid-v5', 'ALE/VideoPinball-v5', 'ALE/Entombed-v5', 'ALE/Centipede-v5', 'ALE/Gopher-v5',
    #     'ALE/Backgammon-v5', 'ALE/Kaboom-v5', 'ALE/Gravitar-v5', 'ALE/Adventure-v5', 'ALE/Seaquest-v5',
    #     'ALE/Turmoil-v5', 'ALE/YarsRevenge-v5', 'ALE/Riverraid-v5', 'ALE/BeamRider-v5', 'ALE/DoubleDunk-v5',
    #     'ALE/UpNDown-v5', 'ALE/TimePilot-v5', 'ALE/BankHeist-v5', 'ALE/Tutankham-v5', 'ALE/Defender-v5',
    #     'ALE/Earthworld-v5', 'ALE/Blackjack-v5', 'ALE/Superman-v5', 'ALE/Klax-v5', 'ALE/Casino-v5',
    #     'ALE/DemonAttack-v5', 'ALE/Zaxxon-v5', 'ALE/HumanCannonball-v5', 'ALE/Hero-v5', 'ALE/Frostbite-v5',
    #     'ALE/PrivateEye-v5', 'ALE/KungFuMaster-v5', 'ALE/Qbert-v5', 'ALE/Surround-v5', 'ALE/Videochess-v5',
    #     'ALE/Krull-v5', 'ALE/Skiing-v5', 'ALE/Darkchambers-v5', 'ALE/Hangman-v5', 'ALE/KeystoneKapers-v5',
    #     'ALE/Freeway-v5', 'ALE/ChopperCommand-v5', 'ALE/FishingDerby-v5', 'ALE/Trondead-v5', 'ALE/CrazyClimber-v5',
    #     'ALE/StarGunner-v5', 'ALE/IceHockey-v5', 'ALE/Amidar-v5', 'ALE/MarioBros-v5', 'ALE/Phoenix-v5',
    #     'ALE/JourneyEscape-v5', 'ALE/Solaris-v5', 'ALE/Atlantis-v5', 'ALE/WordZapper-v5', 'ALE/WizardOfWor-v5',
    #     'ALE/BattleZone-v5', 'ALE/MrDo-v5', 'ALE/Jamesbond-v5', 'ALE/Crossbow-v5', 'ALE/Venture-v5',
    #     'ALE/FlagCapture-v5', 'ALE/BasicMath-v5', 'ALE/Carnival-v5', 'ALE/Bowling-v5', 'ALE/Robotank-v5',
    #     'ALE/Pooyan-v5', 'ALE/Pitfall-v5', 'ALE/Berzerk-v5', 'ALE/Othello-v5', 'ALE/LostLuggage-v5', 'ALE/Assault-v5',
    #     'ALE/HauntedHouse-v5', 'ALE/SirLancelot-v5', 'ALE/SpaceWar-v5', 'ALE/Kangaroo-v5', 'ALE/VideoCheckers-v5',
    #     'ALE/MontezumaRevenge-v5', 'ALE/LaserGates-v5', 'ALE/Atlantis2-v5', 'ALE/Frogger-v5'
    # ]

    env = _get_custom_env(config)
    if env is None:
        env = gym.make(config.environment.name)
        env = ResizeWrapper(env, config.agent.image_shape)
        env = PytorchWrapper(env)
    return Monitor(env)
