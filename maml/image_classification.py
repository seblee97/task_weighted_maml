from .model import MAML, ModelNetwork

class ClassificationMAML(MAML):

    def __init__(self, args):
        self.model = ImageClassificationNetwork(args)
        MAML.__init__(self, args)

    def _generate_batch(self):
        raise NotImplementedError("TODO: different arguments for implementations of abstract classes")

class ImageClassificationNetwork(ModelNetwork):

    def __init__(self):
        ModelNetwork.__init__(self)

    def construct_layers(self):
        in_channels = self.args.in_channels
        num_filters = self.args.num_filters
        filter_size = self.args.filter_size
        pool_size = self.args.pool_size

        self.conv1 = nn.conv2d(in_channels, out_channels=num_filters, kernel_size=filter_size)
        self.conv2 = nn.conv2d(num_filters, out_channels=num_filters, kernel_size=filter_size)
        self.conv3 = nn.conv2d(num_filters, out_channels=num_filters, kernel_size=filter_size)
        self.conv4 = nn.conv2d(num_filters, out_channels=num_filters, kernel_size=filter_size)
        
        self.linear = None #TODO
        self.max_pool = nn.max_pool2d(2)

    def forward(self, x):
        x = F.relu(F.batch_norm(self.conv1(x)))
        x = F.relu(F.batch_norm(self.conv2(x)))
        x = F.relu(F.batch_norm(self.conv3(x)))
        x = F.relu(F.batch_norm(self.conv4(x)))

        x = F.relu(self.linear(x))
        x = self.max_pool(x)
        return x