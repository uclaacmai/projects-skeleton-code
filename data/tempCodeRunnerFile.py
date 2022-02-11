    def augment(self, index):
        image, label = ImageAugment.__getitem__(index)

        self.images.append(self.transform1(image))
        self.labels.append(index)

        self.images.append(self.transform2(image))
        self.labels.append(index)

        self.images.append(self.transform2(image))
        self.labels.append(index)

        self.images.append(self.transform3(image))
        self.labels.append(index)

        self.images.append(self.transform3(image))
        self.labels.append(index)