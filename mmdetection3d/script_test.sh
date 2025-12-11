SAMPLES=$(ls ~/data/nuscenes/samples/LIDAR_TOP/*.pcd.bin | head -10)

# Counter
i=1

# Run inference on each sample
for SAMPLE in $SAMPLES; do
    echo "Processing sample $i: $SAMPLE"

    ((i++))
done

echo "Batch inference complete!"