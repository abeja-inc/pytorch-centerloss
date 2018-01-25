# pytorch-centerloss

## small set of cifar10 experiment

```
python cifar10-ssc.py <expname>
```
where <expname> is any string which will appeared in the "runs" area of in Tensorboard.

You can change centerloss computation scheme just commenting-out this parts;
```
centerloss = ImprovedCenterLoss(10, 0.6)
#centerloss = CenterLoss(10, 10, 0.6)
#centerloss = NullCenterLoss()
```
