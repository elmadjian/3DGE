import matplotlib.pyplot as plt

def plot(filename, variants):
    for v in variants:
        file_to_open = filename + v
        fig, ax = plt.subplots()
        depths, targets, counts = [], [], []
        count = 1
        with open(file_to_open, 'r') as f:
            f.readline()
            depth, v1 = [], 0
            for line in f.readlines():
                if line == '\n':
                    depths.append(depth)
                    depth = []
                    targets.append(float(v1))
                    counts.append(count)
                    count += 1
                    continue
                line = line.replace(',', '.')
                v1, v2 = line.split(' ')
                depth.append(float(v2))
            ax.boxplot(depths)
            ax.plot(counts, targets, 'og-', markersize=7)
            ax.set_xlabel('')
            ax.set_ylabel('depth (m)')
            ax.set_title('Discriminance of estimated depth')
            plt.show()
                


    # ax.legend(shadow=True)
    # ax.set_ylim(0.8, 1)
    # #ax.set_xlim(175,201)
    # ax.set_xlabel('window size')
    # ax.set_ylabel('accuracy')
    # ax.set_title(dataset)
    # plt.show()



if __name__=="__main__":
    filename = "depth_data_"
    variants = ["v1.txt", "v2.txt", "v3.txt", "v4.txt"]
    plot(filename, variants)
