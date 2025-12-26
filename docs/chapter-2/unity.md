---
title: Chapter 5 - Unity Integration and High-Fidelity Rendering
description: "Integrating Unity for high-fidelity rendering and human-robot interaction"
module: 2
chapter: 5
learning_objectives:
  - Understand Unity's role in robotics simulation and visualization
  - Implement Unity for high-fidelity rendering of robotic environments
  - Integrate Unity with ROS 2 for human-robot interaction
  - Create immersive environments for robot simulation
difficulty: advanced
estimated_time: 120
tags:
  - unity
  - rendering
  - simulation
  - hri
  - vr
  - robotics

authors:
  - Textbook Team
prerequisites:
  - Chapter 1: ROS 2 Fundamentals and Architecture
  - Chapter 4: Gazebo Simulation Fundamentals
  - Basic understanding of 3D graphics concepts
---

# Chapter 5: Unity Integration and High-Fidelity Rendering

## Introduction

Unity is a powerful 3D development platform that offers high-fidelity rendering capabilities, making it an excellent choice for creating immersive and visually compelling robotic simulation environments. This chapter explores how to integrate Unity with ROS 2 for advanced robotics applications, focusing on high-fidelity rendering and human-robot interaction.

While Gazebo excels at physics simulation, Unity provides superior visual rendering, real-time graphics, and immersive experiences that are essential for applications requiring photorealistic visualization, virtual reality (VR) interfaces, or sophisticated human-robot interaction (HRI) scenarios.

## Unity in Robotics Context

### Why Unity for Robotics?

Unity offers several advantages for robotics applications:

1. **High-Fidelity Graphics**: Advanced rendering capabilities with physically-based materials, realistic lighting, and post-processing effects
2. **Cross-Platform Support**: Deploy to multiple platforms including VR/AR headsets, mobile devices, and web browsers
3. **Asset Ecosystem**: Extensive marketplace of 3D models, materials, and tools
4. **Real-time Performance**: Optimized for real-time rendering with high frame rates
5. **User Interface Framework**: Robust UI system for creating intuitive interfaces
6. **VR/AR Support**: Native support for virtual and augmented reality applications

### Unity vs. Gazebo

While Gazebo excels at physics simulation, Unity focuses on visual rendering and user experience:

| Aspect | Gazebo | Unity |
|--------|--------|-------|
| Physics | High-fidelity physics engine | Basic physics, mainly for interaction |
| Rendering | Good for robotics visualization | Photorealistic rendering |
| User Interface | Basic GUI | Advanced UI/UX capabilities |
| VR/AR | Limited support | Native VR/AR support |
| Performance | Optimized for simulation | Optimized for real-time graphics |
| Use Case | Physics simulation, testing algorithms | Visualization, HRI, VR/AR |

## Setting Up Unity for Robotics

### Unity ROS Integration Packages

Several packages facilitate ROS integration with Unity:

1. **Unity Robotics Hub**: Official package providing tools and samples
2. **ROS-TCP-Connector**: Enables TCP communication between ROS and Unity
3. **Unity-Robotics-Demo**: Sample projects demonstrating integration

### Basic Setup Process

1. **Install Unity**: Download and install Unity Hub and a compatible Unity version (2021.3 LTS recommended)
2. **Install ROS-TCP-Connector**: Add the package via Unity Package Manager
3. **Configure Network Settings**: Set up TCP communication between Unity and ROS

```csharp
// Example Unity script for ROS communication
using UnityEngine;
using Unity.Robotics.ROSTCPConnector;

public class RobotController : MonoBehaviour
{
    // ROS TCP connector reference
    private RosConnection ros;

    // Robot topic names
    private string robotCmdTopic = "/cmd_vel";
    private string robotStateTopic = "/robot_state";

    void Start()
    {
        // Get reference to ROS connector
        ros = RosConnection.GetOrCreateInstance();

        // Subscribe to robot state topic
        ros.Subscribe<OdometryMsg>(robotStateTopic, RobotStateCallback);
    }

    void RobotStateCallback(OdometryMsg robotState)
    {
        // Update robot position in Unity
        transform.position = new Vector3(
            (float)robotState.pose.pose.position.x,
            (float)robotState.pose.pose.position.y,
            (float)robotState.pose.pose.position.z
        );
    }

    void Update()
    {
        // Send commands to robot based on user input
        if (Input.GetKeyDown(KeyCode.Space))
        {
            var cmd = new TwistMsg();
            cmd.linear.x = 1.0f;
            cmd.angular.z = 0.5f;
            ros.Publish(robotCmdTopic, cmd);
        }
    }
}
```

## High-Fidelity Rendering Techniques

### Physically-Based Rendering (PBR)

Unity's PBR system creates realistic materials that respond to lighting conditions:

```csharp
// Material setup for robot components
public class RobotMaterialSetup : MonoBehaviour
{
    public Material robotBodyMaterial;
    public Material robotJointMaterial;

    void Start()
    {
        // Set robot body material properties
        robotBodyMaterial.SetColor("_BaseColor", Color.gray);
        robotBodyMaterial.SetFloat("_Metallic", 0.7f);
        robotBodyMaterial.SetFloat("_Smoothness", 0.5f);

        // Set joint material properties
        robotJointMaterial.SetColor("_BaseColor", Color.blue);
        robotJointMaterial.SetFloat("_Metallic", 0.9f);
        robotJointMaterial.SetFloat("_Smoothness", 0.8f);
    }
}
```

### Lighting Setup

Proper lighting enhances the realism of robotic simulations:

```csharp
// Lighting configuration script
public class EnvironmentLighting : MonoBehaviour
{
    public Light mainLight;
    public Light[] fillLights;
    public ReflectionProbe reflectionProbe;

    void Start()
    {
        // Configure main directional light
        mainLight.type = LightType.Directional;
        mainLight.intensity = 1.5f;
        mainLight.color = Color.white;
        mainLight.shadows = LightShadows.Soft;

        // Set up fill lights to reduce harsh shadows
        foreach (var light in fillLights)
        {
            light.intensity = 0.3f;
            light.color = Color.gray;
        }

        // Update reflection probe for realistic reflections
        reflectionProbe.RenderProbe();
    }
}
```

### Post-Processing Effects

Enhance visual quality with post-processing:

```csharp
using UnityEngine.Rendering;
using UnityEngine.Rendering.Universal;

public class PostProcessingSetup : MonoBehaviour
{
    public VolumeProfile volumeProfile;

    void Start()
    {
        // Apply post-processing effects
        var bloom = volumeProfile.components.Find<Bloom>();
        if (bloom != null)
        {
            bloom.threshold.value = 1.0f;
            bloom.intensity.value = 0.5f;
        }

        var colorAdjust = volumeProfile.components.Find<ColorAdjustments>();
        if (colorAdjust != null)
        {
            colorAdjust.contrast.value = 10f;
            colorAdjust.saturation.value = 10f;
        }
    }
}
```

## ROS Integration Patterns

### Publisher Pattern in Unity

```csharp
using Unity.Robotics.ROSTCPConnector;
using RosMessageTypes.Sensor;
using RosMessageTypes.Geometry;

public class UnitySensorPublisher : MonoBehaviour
{
    private RosConnection ros;
    public string sensorTopic = "/unity_sensor_data";
    public Camera sensorCamera;

    void Start()
    {
        ros = RosConnection.GetOrCreateInstance();
        InvokeRepeating("PublishSensorData", 0.0f, 0.1f); // Publish at 10 Hz
    }

    void PublishSensorData()
    {
        // Create sensor data message
        ImageMsg sensorMsg = new ImageMsg();

        // Capture image from Unity camera
        Texture2D capturedImage = CaptureCameraImage(sensorCamera);

        // Convert to ROS message format
        sensorMsg.height = (uint)capturedImage.height;
        sensorMsg.width = (uint)capturedImage.width;
        sensorMsg.encoding = "rgb8";
        sensorMsg.is_bigendian = 0;
        sensorMsg.step = (uint)(capturedImage.width * 3); // 3 bytes per pixel (RGB)

        // Convert texture to byte array
        byte[] imageData = capturedImage.EncodeToPNG();
        sensorMsg.data = imageData;

        // Publish message
        ros.Publish(sensorTopic, sensorMsg);
    }

    Texture2D CaptureCameraImage(Camera camera)
    {
        // Capture the camera output to a RenderTexture
        RenderTexture currentRT = RenderTexture.active;
        RenderTexture.active = camera.targetTexture;

        camera.Render();

        Texture2D image = new Texture2D(camera.targetTexture.width, camera.targetTexture.height);
        image.ReadPixels(new Rect(0, 0, camera.targetTexture.width, camera.targetTexture.height), 0, 0);
        image.Apply();

        RenderTexture.active = currentRT;
        return image;
    }
}
```

### Subscriber Pattern in Unity

```csharp
using Unity.Robotics.ROSTCPConnector;
using RosMessageTypes.Geometry;

public class UnityRobotController : MonoBehaviour
{
    private RosConnection ros;
    public string cmdTopic = "/cmd_vel";
    public float linearSpeed = 1.0f;
    public float angularSpeed = 1.0f;

    private float currentLinear = 0.0f;
    private float currentAngular = 0.0f;

    void Start()
    {
        ros = RosConnection.GetOrCreateInstance();
        ros.Subscribe<TwistMsg>(cmdTopic, CommandCallback);
    }

    void CommandCallback(TwistMsg cmd)
    {
        currentLinear = (float)cmd.linear.x;
        currentAngular = (float)cmd.angular.z;
    }

    void Update()
    {
        // Apply robot movement based on received commands
        transform.Translate(Vector3.forward * currentLinear * Time.deltaTime);
        transform.Rotate(Vector3.up, currentAngular * Time.deltaTime);
    }
}
```

## Human-Robot Interaction (HRI) in Unity

### VR Integration for Immersive HRI

Unity provides native support for VR platforms, enabling immersive human-robot interaction:

```csharp
using UnityEngine.XR;
using UnityEngine.XR.Interaction.Toolkit;

public class VRHumanRobotInteraction : MonoBehaviour
{
    public XRNode inputDevice;
    public ActionBasedController controller;
    public GameObject robot;

    void Update()
    {
        // Get input device information
        List<InputDevice> devices = new List<InputDevice>();
        InputDevices.GetDevicesAtXRNode(inputDevice, devices);

        if (devices.Count > 0)
        {
            InputDevice device = devices[0];

            // Check for interaction button press
            if (device.TryGetFeatureValue(CommonUsages.triggerButton, out bool triggerPressed) && triggerPressed)
            {
                // Send command to robot
                SendRobotCommand(robot, "move_forward");
            }

            // Check for grip button press
            if (device.TryGetFeatureValue(CommonUsages.gripButton, out bool gripPressed) && gripPressed)
            {
                // Send different command to robot
                SendRobotCommand(robot, "stop");
            }
        }
    }

    void SendRobotCommand(GameObject robot, string command)
    {
        // Implementation to send command to robot via ROS
        Debug.Log($"Sending command to robot: {command}");
    }
}
```

### Gesture Recognition and Control

```csharp
using UnityEngine;
using System.Collections.Generic;

public class GestureRecognition : MonoBehaviour
{
    public LineRenderer gestureTrail;
    public float gestureThreshold = 0.1f;

    private List<Vector3> gesturePoints = new List<Vector3>();
    private bool isRecording = false;

    void Update()
    {
        if (Input.GetMouseButtonDown(0))
        {
            StartRecordingGesture();
        }

        if (Input.GetMouseButton(0))
        {
            RecordGesturePoint();
        }

        if (Input.GetMouseButtonUp(0))
        {
            ProcessGesture();
        }
    }

    void StartRecordingGesture()
    {
        isRecording = true;
        gesturePoints.Clear();
        gestureTrail.positionCount = 0;
    }

    void RecordGesturePoint()
    {
        if (!isRecording) return;

        Vector3 worldPoint = GetWorldPositionFromMouse();
        gesturePoints.Add(worldPoint);

        gestureTrail.positionCount = gesturePoints.Count;
        gestureTrail.SetPosition(gesturePoints.Count - 1, worldPoint);
    }

    Vector3 GetWorldPositionFromMouse()
    {
        Ray ray = Camera.main.ScreenPointToRay(Input.mousePosition);
        RaycastHit hit;

        if (Physics.Raycast(ray, out hit))
        {
            return hit.point;
        }

        return Vector3.zero;
    }

    void ProcessGesture()
    {
        if (gesturePoints.Count < 5) return; // Minimum points for a valid gesture

        // Analyze gesture pattern and send to robot
        string gestureType = RecognizeGesture(gesturePoints);
        SendGestureToRobot(gestureType);

        isRecording = false;
    }

    string RecognizeGesture(List<Vector3> points)
    {
        // Simple gesture recognition algorithm
        // In practice, you would use more sophisticated pattern matching
        float totalX = 0, totalY = 0;

        for (int i = 1; i < points.Count; i++)
        {
            Vector3 diff = points[i] - points[i-1];
            totalX += Mathf.Abs(diff.x);
            totalY += Mathf.Abs(diff.y);
        }

        if (totalX > totalY)
            return "horizontal";
        else
            return "vertical";
    }

    void SendGestureToRobot(string gestureType)
    {
        // Send gesture to robot via ROS
        Debug.Log($"Gesture recognized: {gestureType}");
    }
}
```

## Environment Creation and Asset Management

### Procedural Environment Generation

```csharp
using UnityEngine;
using System.Collections.Generic;

public class ProceduralEnvironment : MonoBehaviour
{
    public GameObject[] floorTiles;
    public GameObject[] wallPrefabs;
    public GameObject[] obstaclePrefabs;

    [Range(5, 50)]
    public int gridSize = 20;

    void Start()
    {
        GenerateEnvironment();
    }

    void GenerateEnvironment()
    {
        // Create floor grid
        for (int x = 0; x < gridSize; x++)
        {
            for (int z = 0; z < gridSize; z++)
            {
                GameObject tile = Instantiate(
                    floorTiles[Random.Range(0, floorTiles.Length)],
                    new Vector3(x, 0, z),
                    Quaternion.identity
                );
                tile.transform.SetParent(transform);
            }
        }

        // Add random obstacles
        int obstacleCount = (int)(gridSize * gridSize * 0.1f); // 10% of grid
        for (int i = 0; i < obstacleCount; i++)
        {
            int x = Random.Range(1, gridSize - 1);
            int z = Random.Range(1, gridSize - 1);

            Vector3 position = new Vector3(x, 0.5f, z);

            // Check if position is not already occupied
            if (!IsPositionOccupied(position))
            {
                GameObject obstacle = Instantiate(
                    obstaclePrefabs[Random.Range(0, obstaclePrefabs.Length)],
                    position,
                    Quaternion.identity
                );
                obstacle.transform.SetParent(transform);
            }
        }
    }

    bool IsPositionOccupied(Vector3 position)
    {
        Collider[] colliders = Physics.OverlapSphere(position, 0.5f);
        return colliders.Length > 0;
    }
}
```

### Asset Optimization for Performance

```csharp
using UnityEngine;

public class AssetOptimization : MonoBehaviour
{
    public bool useLOD = true;
    public bool useOcclusionCulling = true;
    public bool useOcclusionCullingForRobots = true;

    void Start()
    {
        if (useLOD)
        {
            SetupLODGroups();
        }

        if (useOcclusionCulling)
        {
            SetupOcclusionCulling();
        }
    }

    void SetupLODGroups()
    {
        // Create LOD group for complex robot models
        LODGroup lodGroup = gameObject.AddComponent<LODGroup>();

        LOD[] lods = new LOD[3];

        // LOD 0: High detail (100% screen size)
        Renderer[] highDetailRenderers = GetComponentsInChildren<Renderer>();
        lods[0] = new LOD(1.0f, highDetailRenderers);

        // LOD 1: Medium detail (50% screen size)
        Renderer[] mediumDetailRenderers = GetMediumDetailRenderers();
        lods[1] = new LOD(0.5f, mediumDetailRenderers);

        // LOD 2: Low detail (25% screen size)
        Renderer[] lowDetailRenderers = GetLowDetailRenderers();
        lods[2] = new LOD(0.25f, lowDetailRenderers);

        lodGroup.SetLODs(lods);
        lodGroup.RecalculateBounds();
    }

    void SetupOcclusionCulling()
    {
        // Enable occlusion culling for static objects
        StaticOcclusionCulling.Compute();
    }

    Renderer[] GetMediumDetailRenderers()
    {
        // Return renderers for medium detail level
        // Implementation depends on your specific asset structure
        return new Renderer[0];
    }

    Renderer[] GetLowDetailRenderers()
    {
        // Return renderers for low detail level
        // Implementation depends on your specific asset structure
        return new Renderer[0];
    }
}
```

## Advanced Rendering Techniques

### Real-time Ray Tracing

For high-end applications requiring photorealistic rendering:

```csharp
#if UNITY_EDITOR
using UnityEditor.Rendering;
#endif

public class RayTracingSetup : MonoBehaviour
{
    public bool enableRayTracing = false;

    void Start()
    {
        if (enableRayTracing)
        {
            SetupRayTracing();
        }
    }

    void SetupRayTracing()
    {
        // Configure ray tracing settings
        // Note: Requires compatible hardware and Unity Pro/Enterprise
        RenderPipelineManager.beginCameraRendering += OnBeginCameraRendering;
    }

    void OnBeginCameraRendering(ScriptableRenderContext context, Camera camera)
    {
        // Apply ray tracing effects
        if (enableRayTracing)
        {
            // Enable ray-traced reflections, shadows, etc.
            // Implementation depends on your specific ray tracing solution
        }
    }
}
```

### Multi-Camera Setup for Different Views

```csharp
public class MultiCameraSetup : MonoBehaviour
{
    public Camera mainCamera;
    public Camera robotCamera;
    public Camera overheadCamera;
    public Camera[] sensorCameras;

    [Header("Camera Switching")]
    public KeyCode[] cameraKeys = { KeyCode.Alpha1, KeyCode.Alpha2, KeyCode.Alpha3 };
    private Camera[] cameras;
    private int currentCameraIndex = 0;

    void Start()
    {
        cameras = new Camera[] { mainCamera, robotCamera, overheadCamera };

        // Initialize all cameras
        foreach (var cam in cameras)
        {
            if (cam != null)
            {
                cam.enabled = false;
            }
        }

        // Enable the first camera
        if (cameras.Length > 0 && cameras[0] != null)
        {
            cameras[0].enabled = true;
        }
    }

    void Update()
    {
        // Switch between cameras using number keys
        for (int i = 0; i < cameraKeys.Length && i < cameras.Length; i++)
        {
            if (Input.GetKeyDown(cameraKeys[i]))
            {
                SwitchCamera(i);
            }
        }
    }

    void SwitchCamera(int index)
    {
        // Disable all cameras
        foreach (var cam in cameras)
        {
            if (cam != null)
            {
                cam.enabled = false;
            }
        }

        // Enable selected camera
        if (index >= 0 && index < cameras.Length && cameras[index] != null)
        {
            cameras[index].enabled = true;
            currentCameraIndex = index;
        }
    }
}
```

## Performance Optimization

### Occlusion and Frustum Culling

```csharp
using UnityEngine;

public class PerformanceOptimization : MonoBehaviour
{
    public float updateInterval = 0.1f;
    private float lastUpdateTime = 0f;

    void Update()
    {
        if (Time.time - lastUpdateTime > updateInterval)
        {
            lastUpdateTime = Time.time;
            OptimizeScene();
        }
    }

    void OptimizeScene()
    {
        // Cull objects outside camera view
        Camera mainCamera = Camera.main;
        if (mainCamera == null) return;

        Vector3 cameraPos = mainCamera.transform.position;
        float viewDistance = mainCamera.farClipPlane;

        // Disable objects that are too far away
        GameObject[] allObjects = FindObjectsOfType<GameObject>();
        foreach (GameObject obj in allObjects)
        {
            if (Vector3.Distance(cameraPos, obj.transform.position) > viewDistance * 0.8f)
            {
                obj.SetActive(false);
            }
            else
            {
                obj.SetActive(true);
            }
        }
    }
}
```

### Level of Detail (LOD) Management

```csharp
using UnityEngine;

public class LODManager : MonoBehaviour
{
    [System.Serializable]
    public class LODLevel
    {
        public float screenPercentage;
        public GameObject[] objects;
    }

    public LODLevel[] lodLevels;
    public Camera referenceCamera;

    private int currentLOD = 0;

    void Start()
    {
        if (referenceCamera == null)
            referenceCamera = Camera.main;
    }

    void Update()
    {
        if (referenceCamera == null) return;

        float distance = Vector3.Distance(transform.position, referenceCamera.transform.position);
        float screenPercentage = CalculateScreenPercentage(distance);

        UpdateLOD(screenPercentage);
    }

    float CalculateScreenPercentage(float distance)
    {
        // Calculate what percentage of the screen the object occupies
        float objectSize = Mathf.Max(transform.localScale.x,
                                   Mathf.Max(transform.localScale.y, transform.localScale.z));
        float screenPercentage = (objectSize / distance) * Screen.height;
        return screenPercentage;
    }

    void UpdateLOD(float screenPercentage)
    {
        int newLOD = 0;

        // Find the appropriate LOD level
        for (int i = 0; i < lodLevels.Length; i++)
        {
            if (screenPercentage > lodLevels[i].screenPercentage)
            {
                newLOD = i;
                break;
            }
        }

        // Apply the new LOD if it's different
        if (newLOD != currentLOD)
        {
            SetLOD(newLOD);
            currentLOD = newLOD;
        }
    }

    void SetLOD(int lodIndex)
    {
        // Enable objects for the current LOD
        for (int i = 0; i < lodLevels.Length; i++)
        {
            bool enable = (i == lodIndex);

            if (i < lodLevels.Length)
            {
                foreach (GameObject obj in lodLevels[i].objects)
                {
                    if (obj != null)
                        obj.SetActive(enable);
                }
            }
        }
    }
}
```

## Integration Best Practices

### Communication Architecture

```csharp
using System.Collections.Generic;
using UnityEngine;

public class UnityROSBridge : MonoBehaviour
{
    // Singleton pattern for global access
    public static UnityROSBridge Instance { get; private set; }

    [Header("ROS Settings")]
    public string rosIP = "127.0.0.1";
    public int rosPort = 10000;

    [Header("Topic Configuration")]
    public List<TopicConfig> topicConfigs;

    [System.Serializable]
    public class TopicConfig
    {
        public string topicName;
        public string messageType;
        public bool isPublisher;
        public bool isSubscriber;
    }

    private Dictionary<string, object> topicData = new Dictionary<string, object>();

    void Awake()
    {
        if (Instance == null)
        {
            Instance = this;
            DontDestroyOnLoad(gameObject);
        }
        else
        {
            Destroy(gameObject);
        }
    }

    void Start()
    {
        SetupROSConnection();
        InitializeTopics();
    }

    void SetupROSConnection()
    {
        // Initialize ROS TCP connection
        // Implementation depends on your ROS connector package
        Debug.Log($"Connecting to ROS at {rosIP}:{rosPort}");
    }

    void InitializeTopics()
    {
        foreach (var config in topicConfigs)
        {
            if (config.isPublisher)
            {
                // Setup publisher for this topic
                SetupPublisher(config.topicName, config.messageType);
            }

            if (config.isSubscriber)
            {
                // Setup subscriber for this topic
                SetupSubscriber(config.topicName, config.messageType);
            }
        }
    }

    void SetupPublisher(string topicName, string messageType)
    {
        // Implementation for setting up ROS publisher
        Debug.Log($"Setting up publisher for {topicName} ({messageType})");
    }

    void SetupSubscriber(string topicName, string messageType)
    {
        // Implementation for setting up ROS subscriber
        Debug.Log($"Setting up subscriber for {topicName} ({messageType})");
    }

    public void Publish<T>(string topicName, T message)
    {
        // Publish message to ROS topic
        Debug.Log($"Publishing to {topicName}: {message}");
    }

    public T Subscribe<T>(string topicName)
    {
        // Subscribe to ROS topic and return latest message
        if (topicData.ContainsKey(topicName))
        {
            return (T)topicData[topicName];
        }

        return default(T);
    }
}
```

## Summary

This chapter covered the integration of Unity with ROS 2 for high-fidelity rendering and human-robot interaction:

- Unity's role in robotics simulation and visualization
- Setting up Unity for robotics applications with ROS integration
- High-fidelity rendering techniques using PBR and post-processing
- ROS communication patterns in Unity (publishers and subscribers)
- Human-robot interaction techniques including VR and gesture recognition
- Environment creation and asset optimization
- Advanced rendering techniques and performance optimization
- Best practices for Unity-ROS integration

Unity provides powerful capabilities for creating immersive, visually compelling robotic applications that complement traditional physics simulators like Gazebo.

## Learning Check

After completing this chapter, you should be able to:
- Set up Unity for robotics applications with ROS integration
- Implement high-fidelity rendering techniques in Unity
- Create effective human-robot interaction interfaces
- Design and optimize 3D environments for robotic simulation
- Apply performance optimization techniques for real-time rendering
- Integrate Unity with ROS 2 using appropriate communication patterns