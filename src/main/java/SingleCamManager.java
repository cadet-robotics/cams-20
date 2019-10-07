import edu.wpi.cscore.CvSink;
import edu.wpi.cscore.CvSource;
import edu.wpi.cscore.MjpegServer;
import edu.wpi.cscore.VideoSource;
import edu.wpi.first.cameraserver.CameraServer;
import edu.wpi.first.networktables.NetworkTable;
import edu.wpi.first.networktables.NetworkTableEntry;
import edu.wpi.first.networktables.NetworkTableInstance;
import org.opencv.core.*;
import org.opencv.imgproc.Imgproc;

import java.util.ArrayList;

public class SingleCamManager extends Thread {
    private NetworkTableEntry testEntry;
    private NetworkTable targetsTable;
    private NetworkTableEntry weightsEntry;

    private static CvSink cam;

    private static final double H_FOV = 49;
    private static final double V_FOV = 49;

    private CameraServer camServer;
    private String name;

    private RectCmpWeights weights;

    public SingleCamManager(CameraServer camServerIn, NetworkTableInstance ntIn, VideoSource camIn) {
        camServer = camServerIn;

        cam = camServer.getVideo(camIn);
        name = camIn.getName();

        NetworkTable t = ntIn.getTable("ShuffleBoard").getSubTable("cam-vals-" + name);
        targetsTable = t.getSubTable("targets");
        testEntry = t.getEntry("test");
        weightsEntry = t.getEntry("weights");

        weights = new RectCmpWeights();
        weights.update(weightsEntry);
    }

    private static final Scalar FILTER_LOW = new Scalar(0, 75, 0);
    private static final Scalar FILTER_HIGH = new Scalar(255, 255, 255);

    private static final int BLUR_THRESH = 60;

    private void updateTime() {
        testEntry.setNumber(System.currentTimeMillis());
    }

    private void updateWeights() {
        weights.update(weightsEntry);
    }

    public void run() {
        CvSource originalOut = camServer.putVideo(name + "_original", 320, 240);
        MjpegServer s1 = new MjpegServer(name + "_serv_original", 8089);
        s1.setSource(originalOut);

        CvSource shapesOut = camServer.putVideo(name + "_shapes", 320, 240);
        MjpegServer s2 = new MjpegServer(name + "_serv_shapes", 8090);
        s2.setSource(shapesOut);

        Mat camFrame = new Mat();
        Mat workingFrame = new Mat();
        Mat lineMap = new Mat();
        while (!Thread.interrupted()) {
            ArrayList<MatOfPoint> contours = new ArrayList<>(); // shapes

            cam.grabFrame(camFrame);
            if (!camFrame.empty()) {
                updateTime();
                updateWeights();

                originalOut.putFrame(camFrame);
                Imgproc.cvtColor(camFrame, workingFrame, Imgproc.COLOR_BGR2HLS);                                            // Change color scheme from BGR to HSL
                Core.inRange(workingFrame, FILTER_LOW, FILTER_HIGH, workingFrame);                                                // Filter colors with <250 lightness
                Imgproc.cvtColor(workingFrame, workingFrame, Imgproc.COLOR_GRAY2BGR);
                Imgproc.GaussianBlur(workingFrame, workingFrame, new Size(5, 5), 0);                                            // Blur
                Imgproc.threshold(workingFrame, workingFrame, BLUR_THRESH, 255, Imgproc.THRESH_BINARY);                            // Turn colors <60 black, >=60 white
                Imgproc.cvtColor(workingFrame, workingFrame, Imgproc.COLOR_BGR2GRAY);
                Imgproc.findContours(workingFrame, contours, workingFrame, Imgproc.RETR_LIST, Imgproc.CHAIN_APPROX_SIMPLE);    // Find shapes

                lineMap.create(camFrame.size(), camFrame.type());
                VisionCalcs.wipe(lineMap, new Scalar(0, 0, 0));
                Imgproc.drawContours(lineMap, contours, -1, VisionCalcs.COLOR_WHITE);

                int[] matches = VisionCalcs.pairUp(contours, weights);

                /*
                for (int i = 0; i < matches.length; i++) {
                    if ((matches[i] != -1) && (matches[i] > i)) {
                        System.out.println(VisionCalcs.getRectPairScore(contours.get(i), contours.get(matches[i]), weights));
                    }
                }
                */

                RotatedRect[] rects = new RotatedRect[contours.size()];
                for (int i = 0; i < rects.length; i++) rects[i] = Imgproc.minAreaRect(new MatOfPoint2f(contours.get(i).toArray()));
                for (int i = 0; i < rects.length; i++) {
                    VisionCalcs.drawRectangle(lineMap, rects[i], VisionCalcs.COLOR_RED);
                    if ((matches[i] == -1) || (matches[i] < i)) continue;
                    Imgproc.line(lineMap, rects[i].center, rects[matches[i]].center, VisionCalcs.COLOR_RED);
                }
                shapesOut.putFrame(lineMap);

                // Upload data
                for (int i = 0; i < matches.length; i++) {
                    if (i < matches[i]) {
                        VisionCalcs.pack(targetsTable.getEntry(Integer.toString(i)), H_FOV, V_FOV, camFrame.size(), rects[i], rects[matches[i]], (new RectPair(contours.get(i), contours.get(matches[i]), weights)).getScore());
                    }
                }
            } else System.err.println("Failed to get stream");
        }

        s1.close();
        s2.close();
    }
}