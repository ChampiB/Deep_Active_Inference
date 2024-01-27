import torch


class Device:
    """
    Singleton to access the type of device to use, i.e. GPU or CPU.
    """

    # Specify the device on which models and tensors must be sent (i.e., None, cpu or cuda).
    device_name = "cuda"

    @staticmethod
    def get():
        """
        Getter
        :return: the device on which computation should be performed
        """

        # If the device was not provided by the user:
        # - select cuda if cuda is available
        # - select cpu otherwise
        if Device.device_name is None:
            Device.device_name = 'cuda' if torch.cuda.is_available() and torch.cuda.device_count() >= 1 else 'cpu'

        # Create the device.
        return torch.device(Device.device_name)

    @staticmethod
    def send(models):
        """
        Send the models to the device, i.e., gpu or cpu
        :param models: the list of model to send to the device
        """
        device = Device.get()
        for model in models:
            model.to(device)
