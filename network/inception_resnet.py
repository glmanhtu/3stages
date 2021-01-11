from facenet_pytorch.models.inception_resnet_v1 import InceptionResnetV1
from torch import nn
from torch.nn import functional as F

from utils.constants import device as device_id


class InceptionResnetCustom(InceptionResnetV1):
    def __init__(self, pretrained=None, classify=False, num_classes=None, dropout_prob=0.6, device=None):
        """
        The aim of this class is to modify the forward method, for extracting the embedding features
        Without having to modify the original class
        """
        super(InceptionResnetCustom, self).__init__(pretrained, classify, num_classes, dropout_prob, device)

    def forward(self, x):
        """
        @rtype: tensor object with shape (batch_size, 1792)
        """
        x = self.conv2d_1a(x)
        x = self.conv2d_2a(x)
        x = self.conv2d_2b(x)
        x = self.maxpool_3a(x)
        x = self.conv2d_3b(x)
        x = self.conv2d_4a(x)
        x = self.conv2d_4b(x)
        x = self.repeat_1(x)
        x = self.mixed_6a(x)
        x = self.repeat_2(x)
        x = self.mixed_7a(x)
        x = self.repeat_3(x)
        x = self.block8(x)
        x = self.avgpool_1a(x)
        x = x.view(x.shape[0], -1)

        return F.normalize(x, p=2, dim=1)


class InceptionHeatMap(InceptionResnetV1):

    def __init__(self, num_aus, device, embedded=False, dropout=0.8):
        """
        The aim of this class is to modify the forward method, for extracting the embedding features
        Without having to modify the original class
        """
        super(InceptionHeatMap, self).__init__('vggface2', classify=False, device=device, num_classes=1,
                                               dropout_prob=dropout)
        del self.mixed_7a
        del self.repeat_3
        del self.block8
        del self.avgpool_1a
        del self.last_linear
        del self.last_bn
        del self.logits

        self.up_scaling_1 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=896, padding=1, out_channels=160, kernel_size=(4, 4), stride=2),
            nn.ReLU()
        )

        self.up_scaling_2 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=160, padding=1, out_channels=80, kernel_size=(4, 4), stride=2),
            nn.ReLU()
        )

        self.up_scaling_3 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=80, padding=1, out_channels=40, kernel_size=(4, 4), stride=2),
            nn.ReLU()
        )

        self.final = nn.Conv2d(in_channels=40, out_channels=num_aus, kernel_size=(1, 1))
        self.dropout_2d = nn.Dropout2d(p=0.6)
        self.to(device)
        self.embedded = embedded

    def forward(self, x):
        x = self.conv2d_1a(x)
        x = self.conv2d_2a(x)
        x = self.conv2d_2b(x)
        x = self.maxpool_3a(x)
        x = self.conv2d_3b(x)
        x = self.conv2d_4a(x)
        x = self.conv2d_4b(x)
        x = self.repeat_1(x)
        x = self.mixed_6a(x)
        x = self.repeat_2(x)
        # x = self.mixed_7a(x)
        # x = self.repeat_3(x)
        # x = self.block8(x)
        if self.embedded:
            x = x.reshape(x.shape[0], -1)
            return x
        x = self.up_scaling_1(x)
        x = self.up_scaling_2(x)
        x = self.up_scaling_3(x)
        x = self.dropout_2d(x)
        x = self.final(x)
        return x


def get_pretrained_facenet(classify=True, num_classes=1, pretrained='vggface2'):
    if not classify:
        model = InceptionResnetCustom(pretrained=pretrained, classify=True, device=device_id,
                                      num_classes=num_classes)
    else:
        model = InceptionResnetV1(pretrained=pretrained, classify=classify, device=device_id, num_classes=num_classes)

    return model
