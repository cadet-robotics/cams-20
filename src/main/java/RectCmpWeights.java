import edu.wpi.first.networktables.NetworkTableEntry;

public class RectCmpWeights {
    public double areaWeight;
    public double angleWeight;
    public double areaDiffWeight;
    public double simWeight;

    public RectCmpWeights() {
        this(5, 1, 0.5, 1);
    }

    public RectCmpWeights(double areaWeightIn, double angleWeightIn, double areaDiffWeightIn, double simWeightIn) {
        areaWeight = areaWeightIn;
        angleWeight = angleWeightIn;
        areaDiffWeight = areaDiffWeightIn;
        simWeight = simWeightIn;
    }

    public void update(NetworkTableEntry e) {
        double[] data = e.getDoubleArray((double[]) null);
        if (data != null && data.length >= 4) {
            areaWeight = data[0];
            angleWeight = data[1];
            areaDiffWeight = data[2];
            simWeight = data[3];
        } else {
            e.setDoubleArray(new double[] {areaWeight, angleWeight, areaDiffWeight, simWeight});
        }
    }
}