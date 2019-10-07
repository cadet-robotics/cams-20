import org.opencv.core.*;
import org.opencv.imgproc.Imgproc;

public class RectPair {
    public static final double MIN_RECT_AREA = 50;

    private double score;
    private RotatedRect r1;
    private RotatedRect r2;

    public RectPair(MatOfPoint p1, MatOfPoint p2, RectCmpWeights w) {
        r1 = normalizeRect(Imgproc.minAreaRect(new MatOfPoint2f(p1.toArray())));
        r2 = normalizeRect(Imgproc.minAreaRect(new MatOfPoint2f(p2.toArray())));
        double area1 = r1.size.area();
        if (area1 < MIN_RECT_AREA) {
            score = 0;
        } else {
            double area2 = r2.size.area();
            if (area2 < MIN_RECT_AREA) {
                score = 0;
            } else {
                // 0 to 1
                double areaScore = ((Imgproc.contourArea(p1) / area1) + (Imgproc.contourArea(p2) / area2)) / 2;
                // 0 to 1
                double areaDiffScore = (area2 > area1) ? (area1 / area2) : (area2 / area1);

                RotatedRect t;
                if ((r1.center.x > r2.center.x) || ((r1.center.x == r2.center.x) && (r1.hashCode() > r2.hashCode()))) {
                    t = r1;
                    r1 = r2;
                    r2 = t;
                }

                Point across = new Point(r2.center.x - r1.center.x, r2.center.y - r1.center.y);
                double angleAcross = Math.atan2(across.y, across.x); // IN RADIANS
                if (angleAcross < 0) angleAcross = angleAcross + 2 * Math.PI;

                double angle1 = r1.angle * Math.PI / 180;
                double angle2 = r2.angle * Math.PI / 180;

                double angleAcrossPerp = angleAcross + (Math.PI / 2);

                double angleDiff1 = JavaIsCancerChangeMyMind.moduloIsCancer(angle1 - angleAcrossPerp, Math.PI * 2);
                if (angleDiff1 > Math.PI) angleDiff1 = Math.PI * 2 - angleDiff1;
                double angleDiff2 = JavaIsCancerChangeMyMind.moduloIsCancer(angle2 - angleAcrossPerp, Math.PI * 2);
                if (angleDiff2 > Math.PI) angleDiff2 = Math.PI * 2 - angleDiff2;

                if ((angleDiff1 > (Math.PI / 1.5)) || (angleDiff2 > (Math.PI / 1.5))) {
                    score = 0;
                } else {
                    // 0 to 1
                    double angleScore = (angleDiff1 - angleDiff2) / Math.PI;
                    if (angleScore < 0) angleScore = -angleScore;
                    angleScore = 1 - angleScore;

                    // 0 to 1
                    double simScore = angleScore - areaDiffScore;
                    if (simScore < 0) simScore = -simScore;
                    simScore = 1 - simScore;

                    double comb = w.areaWeight + w.angleWeight + w.areaDiffWeight + w.simWeight;
                    if (comb == 0) {
                        score = 0;
                    } else {
                        score = (areaScore + angleScore + areaDiffScore + simScore) / comb;
                    }
                }
            }
        }
    }

    /**
     * Returns a duplicate of a RotatedRect, but with:
     *
     * * height > width
     * * angle >= 270 or angle <= 90
     *
     * @param in The input rectangle
     * @return The "normalized" rectangle
     */
    @SuppressWarnings("SuspiciousNameCombination")
    public static RotatedRect normalizeRect(RotatedRect in) {
        double angle = in.angle;
        double height = in.size.height;
        double width = in.size.width;
        Size newSize;
        if (height > width) {
            newSize = new Size(height, width);
            angle = (angle + 90) % 360;
        } else {
            newSize = in.size;
        }
        if ((angle > 90) && (angle < 270)) {
            angle = (angle + 180) % 360;
        }
        return new RotatedRect(in.center, newSize, angle);
    }

    public RotatedRect getR1() {
        return r1;
    }

    public RotatedRect getR2() {
        return r2;
    }

    public double getScore() {
        return score;
    }

    @Override
    public int hashCode() {
        int v = 0;
        v = v * 31 + r1.hashCode();
        v = v * 31 + r2.hashCode();
        v = v * 31 + ((int) (score * 32));
        return v;
    }

    @Override
    public boolean equals(Object obj) {
        if (obj instanceof RectPair) {
            RectPair other = (RectPair) obj;
            return r1.equals(other.r1) && r2.equals(other.r2) && score == score;
        } else {
            return this == obj;
        }
    }
}