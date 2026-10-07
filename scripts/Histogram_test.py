from Histogram import Process
import matplotlib.pyplot as plt


if __name__ == '__main__':
    # Quick sanity check over the BDS500 folder

    
    for name, gray, hist, pdf in Process.iter_dataset_images("data/BDS500"):
        fig, axes = plt.subplots(1, 2, figsize=(8, 4))
        axes[0].imshow(gray, cmap="gray", vmin=0,vmax=255)
        axes[0].set_title(name+"GreyScale")
        axes[0].axis("off")

        axes[1].plot(hist)
        axes[1].set_title(name +"Histogram")
        axes[1].set_xlabel("Intensity level")
        axes[1].set_ylabel("Pixel Count")

        plt.tight_layout()
        print(f"{name:10s} shape={gray.shape}  sum(pdf)={pdf.sum():.4f}")
    plt.show()

    