import QtQuick 2.9
import QtQuick.Window 2.3
import QtQuick.Controls 2.2
import QtQuick.Controls.Universal 2.2
import QtGraphicalEffects 1.0
import QtQuick.Dialogs 1.2
import QtQuick.Layouts 1.0

Window {
    id: mainWindow
    visible: true
    width: 1050
    height: 500
    color: "#202020"
    title: qsTr("3D Gaze AR Eye Tracker")
    Universal.theme: Universal.Dark
    Universal.accent: Universal.Lime
    // @disable-check M16
    onClosing: {
        console.log("closing window");
        calibHMD.save_session();
        if (leftEyeGroup.video || rightEyeGroup.video) {
            camManager.stop_cameras(true);
        } else {
            camManager.stop_cameras(false);
        }
    }

    /*
    Left Eye Camera
    ---------------
    */
    GroupBox {
        id: leftEyeGroup
        x: 30
        y: 151
        width: 480
        height: 320
        label: Text {
            id: leftEyeTitle
            x: 7
            y: 5
            color: "#ffffff"
            text: "Left Eye Camera"
            z: 1
            font.weight: Font.Light
        }
        property bool video: false

        Image {
            id: leftIcon3d
            z: 2
            x: 256
            y: 174
            width: 35
            height: 35
            source: "../imgs/reload-icon.png"
            fillMode: Image.PreserveAspectFit
            opacity: 1
            anchors.right: parent.right
            anchors.bottom: parent.bottom

            MouseArea {
                id: leftIcon3dButton
                hoverEnabled: true
                cursorShape: Qt.PointingHandCursor
                anchors.fill: parent
                onClicked: {
                    leftEyeCam.reset();
                }
            }
        }

        ComboBox {
            id: leftEyeBox
            currentIndex: 0
            z: 1
            x: 308
            y: -12
            height: 28
            model: camManager.camera_list
            onActivated:  {
                if (textAt(index) === "File...") {
                    leftEyeFileDialog.visible = true;
                }
                else if (textAt(index) === "No feed") {
                    leftEyeGroup.video?
                                camManager.stop_leye_cam(true):
                                camManager.stop_leye_cam(false);
                    leyeImage.source = "../imgs/novideo.png";
                }
                else {
                    leftEyeGroup.video = false;
                    camManager.set_camera_source(leftEyeTitle.text, textAt(index));
                    activate_config(leftEyeDisabledOverlay, prefLeftEyeImg);
                    enable_functions();
                }
            }
//            Component.onCompleted: {
//                camManager.set_camera_last_session(leftEyeTitle.text)
//            }
        }

        FileDialog {
            id: leftEyeFileDialog
            title: "Please, select a video file"
            folder: shortcuts.home
            visible: false
            nameFilters: ["Video files (*.avi, *.mkv, *.mpeg, *.mp4)", "All files (*)"]
            onAccepted: {
                var file = leftEyeFileDialog.fileUrl.toString();
                var suffix = file.substring(file.indexOf("/")+2);
                camManager.load_video(leftEyeTitle.text, suffix);
                leftEyeGroup.video = true;
                playImg.enabled = true;
            }
            onRejected: {
                leftEyeBox.currentIndex = 0;
            }
        }

        Image {
            id: leyeImage
            property bool counter: false
            anchors.rightMargin: -10
            anchors.leftMargin: -10
            anchors.bottomMargin: -10
            anchors.topMargin: -10
            source: "../imgs/novideo.png"
            anchors.fill: parent
            fillMode: Image.Stretch
            cache: false

            signal updateImage()
            Component.onCompleted: leftEyeCam.update_image.connect(updateImage);

            Connections {
                target: leyeImage
                onUpdateImage: {
                    leyeImage.counter = !leyeImage.counter; //hack to force update
                    leyeImage.source = "image://leyeimg/eye" + leyeImage.counter;
                }
            }
        }

    }

    /*
    Right Eye Camera
    ----------------
    */
    GroupBox {
        id: rightEyeGroup
        x: 536
        y: 151
        width: 480
        height: 320
        visible: true
        label: Text {
            id:rightEyeTitle
            x: 7
            y: 5
            color: "#ffffff"
            text: "Right Eye Camera"
            z: 1
            font.weight: Font.Light
        }
        property bool video: false



        Image {
            id: rightIcon3d
            z: 2
            x: 261
            y: 174
            width: 35
            height: 35
            source: "../imgs/reload-icon.png"
            fillMode: Image.PreserveAspectFit
            opacity: 1
            anchors.right: parent.right
            anchors.bottom: parent.bottom

            MouseArea {
                id: rightIcon3dButton
                hoverEnabled: true
                cursorShape: Qt.PointingHandCursor
                anchors.fill: parent
                onClicked: {
                    rightEyeCam.reset();
                }
            }
        }

        ComboBox {
            id: rightEyeBox
            currentIndex: 0
            z: 1
            x: 315
            y: -12
            height: 28
            model: camManager.camera_list
            onActivated:  {
                if (textAt(index) === "File...") {
                    rightEyeFileDialog.visible = true;
                }
                else if (textAt(index) === "No feed") {
                    rightEyeGroup.video?
                                camManager.stop_reye_cam(true):
                                camManager.stop_reye_cam(false);
                    reyeImage.source = "../imgs/novideo.png";
                }
                else {
                    rightEyeGroup.video = false;
                    camManager.set_camera_source(rightEyeTitle.text, textAt(index));
                    activate_config(rightEyeDisabledOverlay, prefRightEyeImg);
                    enable_functions();
                }
            }
//            Component.onCompleted: {
//                camManager.set_camera_last_session(rightEyeTitle.text)
//            }
        }

        FileDialog {
            id: rightEyeFileDialog
            title: "Please, select a video file"
            folder: shortcuts.home
            visible: false
            nameFilters: ["Video files (*.avi, *.mkv, *.mpeg, *.mp4)", "All files (*)"]
            onAccepted: {
                var file = rightEyeFileDialog.fileUrl.toString();
                var suffix = file.substring(file.indexOf("/")+2);
                camManager.load_video(rightEyeTitle.text, suffix);
                rightEyeGroup.video = true;
                playImg.enabled = true;
            }
            onRejected: {
                rightEyeBox.currentIndex = 0;
            }
        }

        Image {
            id: reyeImage
            property bool counter: false
            anchors.rightMargin: -11
            anchors.leftMargin: -9
            anchors.bottomMargin: -10
            anchors.topMargin: -10
            source: "../imgs/novideo.png"
            anchors.fill: parent
            fillMode: Image.Stretch
            cache: false

            signal updateImage()
            x: -10
            Component.onCompleted: rightEyeCam.update_image.connect(updateImage);

            Connections {
                target: reyeImage
                onUpdateImage: {
                    reyeImage.counter = !reyeImage.counter; //hack to force update
                    reyeImage.source = "image://reyeimg/eye" + reyeImage.counter;
                }
            }
        }


    }


    //Helper functions
    //----------------
    function update_comboboxes(uid, camType) {
        uid.comboFrameRate.model = camType.fps_list;
        uid.comboResolution.model = camType.modes_list;
        uid.comboFrameRate.currentIndex = camType.current_fps_index;
        uid.comboResolution.currentIndex = camType.current_res_index;
        uid.dialGamma.value = camType.gamma_state;
        uid.switchColor.position = camType.color_state;
        uid.switchFlip.position = camType.flip_state;
    }

    function activate_dropdown(uid_active, uid2) {
        uid_active.enabled = !uid_active.enabled;
        uid_active.opacity = !uid_active.opacity;
        uid2.enabled = false;
        uid2.opacity = 0;
    }

    function activate_config(overlay, prefImg) {
        overlay.enabled = false;
        overlay.opacity = 0;
        prefImg.enabled = true;
    }

    function enable_functions() {
        if (prefLeftEyeImg.enabled && prefRightEyeImg.enabled) {
            menu.enabled = true;
            menuDisabledOverlay.opacity = 0;
            calibration.enabled = true;
            calibrationDisabledOverlay.opacity = 0;
            recordImg.enabled = true;
            recordDisabledOverlay.opacity = 0;
        }
    }

    function activate_HMD_calibration() {
        calibHMDitem.visible = true;
        calibHMDitem.keyListenerHMD.focus = true;
    }

    /*
    CALIBRATION CONTOL
    ------------------ */

    GroupBox {
        id: playbackSettings
        x: 400
        y: 25
        width: 211
        height: 110
        label: Text {
            color:"gray"
            text:"Playback"
        }

        ColumnLayout {
            x: 18
            y:0
            Layout.fillHeight: false
            Layout.fillWidth: false
            Layout.alignment: Qt.AlignHCenter | Qt.AlignVCenter

            Text {
                id: playLabel
                text: qsTr("Play")
                color: "white"
                horizontalAlignment: Text.AlignHCenter
            }
            Image {
                id: playImg
                sourceSize.width: 50
                sourceSize.height: 50
                fillMode: Image.PreserveAspectFit
                Layout.preferredHeight: 50
                Layout.preferredWidth: 50
                source: "../imgs/play.png"
                enabled: false
                z:1
                states: [
                    State {
                        name: "stalled"
                        PropertyChanges {
                            target: playImg
                            source: "../imgs/play.png"
                        }
                    },
                    State {
                        name: "playing"
                        PropertyChanges {
                            target: playImg
                            source: "../imgs/play.png"
                        }
                    },
                    State {
                        name: "paused"
                        PropertyChanges {
                            target: playImg
                            source: "../imgs/pause.png"
                        }
                    }
                ]
                Component.onCompleted: state = "stalled";

                ColorOverlay {
                    id: playDisabledOverlay
                    anchors.fill: playImg
                    source: playImg
                    color: "#555555"
                    opacity: 1
                }

                ColorOverlay {
                    id: playOverlay
                    anchors.fill: playImg
                    source: playImg
                    color: "white"
                    opacity: 0
                }

                MouseArea {
                    id: playBtn
                    hoverEnabled: true
                    cursorShape: Qt.PointingHandCursor
                    anchors.fill: parent
                    onEntered: {
                        playOverlay.opacity = 1
                    }
                    onExited: {
                        playOverlay.opacity = 0
                    }
                    onClicked: {
                        if (playImg.state == "stalled" || playImg.state == "paused") {
                            camManager.play_cams(leftEyeGroup.video, rightEyeGroup.video);
                            playImg.state = "playing";
                        }
                        else if (playImg.state == "playing") {
                            camManager.pause_cams(leftEyeGroup.video, rightEyeGroup.video);
                            playImg.state = "paused";
                        }
                    }
                }
            }
        }

        ColumnLayout {
            x: 109
            y: 0
            Layout.fillHeight: false
            Text {
                id: recordLabel
                x: 10
                color: "#ffffff"
                text: qsTr("Record")
                horizontalAlignment: Text.AlignHCenter
            }

            Image {
                id: recordImg
                fillMode: Image.PreserveAspectFit
                enabled: false
                z: 1
                sourceSize.width: 50
                source: "../imgs/record.png"
                Layout.preferredHeight: 50
                sourceSize.height: 50
                Layout.preferredWidth: 50
                states: [
                    State {
                        name: "stalled"
                        PropertyChanges {
                            target: recordImg
                            source: "../imgs/record.png"
                        }
                    },
                    State {
                        name: "recording"
                        PropertyChanges {
                            target: recordImg
                            source: "../imgs/record.png"
                        }
                    }
                ]
                Component.onCompleted: state = "stalled";
                ColorOverlay {
                    id: recordDisabledOverlay
                    color: "#555555"
                    source: recordImg
                    anchors.fill: recordImg
                    opacity: 1
                }

                ColorOverlay {
                    id: recordOverlay
                    color: "#ffffff"
                    source: recordImg
                    anchors.fill: recordImg
                    opacity: 0
                }

                MouseArea {
                    id: recordBtn
                    hoverEnabled: true
                    anchors.fill: parent
                    cursorShape: Qt.PointingHandCursor
                    onEntered: {
                        recordOverlay.opacity = 1
                    }
                    onExited: {
                        recordOverlay.opacity = 0
                    }
                    onClicked: {
                        if (recordImg.state == "stalled") {
                            recordImg.state = "recording";
                            recordOverlay.opacity = 1;
                        }
                        else if (recordImg.state == "recording") {
                            recordImg.state = "stalled";
                            recordOverlay.opacity = 0;
                        }
                        camManager.toggle_recording();
                    }
                }
            }
            Layout.alignment: Qt.AlignHCenter | Qt.AlignVCenter
            Layout.fillWidth: false
        }
    }

    GroupBox {
        id: cameraSettings
        x: 645
        y: 25
        width: 223
        height: 110
        visible: true
        label: Text {
            color:"gray"
            text:"Camera Settings"
        }

        ColumnLayout{
            x: 13
            y: 0
            width: 65
            height: 60
            Layout.alignment: Qt.AlignHCenter | Qt.AlignVCenter

            Text {
                id: prefLeftLabel
                text: qsTr("Left Eye")
                color: "white"
                horizontalAlignment: Text.AlignHCenter
            }
            Image {
                id: prefLeftEyeImg
                sourceSize.height: 45
                sourceSize.width: 45
                fillMode: Image.PreserveAspectFit
                Layout.preferredHeight: 50
                Layout.preferredWidth: 50
                source: "../imgs/eye-left.png"
                enabled: false
                z:1

                ColorOverlay {
                    id: leftEyeDisabledOverlay
                    anchors.fill: prefLeftEyeImg
                    source: prefLeftEyeImg
                    color: "#555555"
                    opacity: 1
                }

                ColorOverlay {
                    id: leftEyeOverlayImg
                    anchors.fill: prefLeftEyeImg
                    source: prefLeftEyeImg
                    color: "white"
                    opacity: 0
                }

                MouseArea {
                    id: prefLeftEyeBtn
                    hoverEnabled: true
                    cursorShape: Qt.PointingHandCursor
                    anchors.fill: parent
                    onEntered: {
                        leftEyeOverlayImg.opacity = 1
                    }
                    onExited: {
                        leftEyeOverlayImg.opacity = 0
                    }
                    onClicked: {
                        update_comboboxes(leftEyePrefDropdown, leftEyeCam);
                        activate_dropdown(leftEyePrefDropdown, rightEyePrefDropdown);
                    }
                }
                Dropdown {
                    id:leftEyePrefDropdown
                    x: -143
                    y: 49
                    enabled: false;
                    opacity: 0;
                    comboFrameRate.onActivated: {
                        leftEyeCam.set_mode(comboFrameRate.currentText, comboResolution.currentText);
                        update_comboboxes(leftEyePrefDropdown, leftEyeCam);
                    }
                    comboResolution.onActivated: {
                        leftEyeCam.set_mode(comboFrameRate.currentText, comboResolution.currentText);
                        update_comboboxes(leftEyePrefDropdown, leftEyeCam);
                    }
                    dialGamma.onMoved: {
                        leftEyeCam.set_gamma(dialGamma.value);
                    }
                    switchColor.onToggled: {
                        leftEyeCam.set_color(switchColor.position);
                    }
                    switchFlip.onToggled: {
                        leftEyeCam.flip_image(switchFlip.position);
                    }
                }
            }
        }


        ColumnLayout {
            x: 127
            y: 0
            width: 65
            height: 60
            Layout.alignment: Qt.AlignHCenter | Qt.AlignVCenter

            Text {
                id: prefRightLabel
                text: qsTr("Right Eye")
                color:"white"
                horizontalAlignment: Text.AlignHCenter
            }
            Image {
                id: prefRightEyeImg
                sourceSize.width: 45
                sourceSize.height: 45
                fillMode: Image.PreserveAspectFit
                Layout.preferredHeight: 50
                Layout.preferredWidth: 50
                source: "../imgs/eye-right.png"
                enabled: false
                z:1

                ColorOverlay {
                    id: rightEyeDisabledOverlay
                    anchors.fill: prefRightEyeImg
                    source: prefRightEyeImg
                    color: "#555555"
                    opacity: 1
                }

                ColorOverlay {
                    id: rightEyeOverlayImg
                    anchors.fill: prefRightEyeImg
                    source: prefRightEyeImg
                    color: "white"
                    opacity: 0
                }

                MouseArea {
                    id: prefRighttEyeBtn
                    hoverEnabled: true
                    cursorShape: Qt.PointingHandCursor
                    anchors.fill: parent
                    onEntered: {
                        rightEyeOverlayImg.opacity = 1
                    }
                    onExited: {
                        rightEyeOverlayImg.opacity = 0
                    }
                    onClicked: {
                        update_comboboxes(rightEyePrefDropdown, rightEyeCam);
                        activate_dropdown(rightEyePrefDropdown, leftEyePrefDropdown);
                    }
                }
                Dropdown {
                    id:rightEyePrefDropdown
                    x: -143
                    y: 49
                    enabled: false;
                    opacity: 0;
                    comboFrameRate.onActivated: {
                        rightEyeCam.set_mode(comboFrameRate.currentText, comboResolution.currentText);
                        update_comboboxes(rightEyePrefDropdown, rightEyeCam);
                    }
                    comboResolution.onActivated: {
                        rightEyeCam.set_mode(comboFrameRate.currentText, comboResolution.currentText);
                        update_comboboxes(rightEyePrefDropdown, rightEyeCam);
                    }
                    dialGamma.onMoved: {
                        rightEyeCam.set_gamma(dialGamma.value);
                    }
                    switchColor.onToggled: {
                        rightEyeCam.set_color(switchColor.position);
                    }
                    switchFlip.onToggled: {
                        rightEyeCam.flip_image(switchFlip.position);
                    }
                }
            }
        }
    }

    GroupBox {
        id: calibrationSettings
        x: 30
        y: 25
        width: 336
        height: 110
        label: Text {
            color:"gray"
            text:"Calibration Settings"
        }
        ColumnLayout {
            x: 12
            y:0
            Layout.fillHeight: false
            Layout.fillWidth: false
            Layout.alignment: Qt.AlignHCenter | Qt.AlignVCenter

            Text {
                id: calibrationLabel
                text: qsTr("Calibrate")
                color: "white"
                horizontalAlignment: Text.AlignHCenter
            }
            Image {
                id: calibration
                sourceSize.width: 50
                sourceSize.height: 50
                fillMode: Image.PreserveAspectFit
                Layout.preferredHeight: 50
                Layout.preferredWidth: 50
                source: "../imgs/calibration.png"
                //enabled: false <-- turned on for DEBUG!
                z:1

                ColorOverlay {
                    id: calibrationDisabledOverlay
                    anchors.fill: calibration
                    source: calibration
                    color: "#555555"
                    opacity: 1
                }

                ColorOverlay {
                    id: calibrationOverlay
                    anchors.fill: calibration
                    source: calibration
                    color: "white"
                    opacity: 0
                }

                MouseArea {
                    id: calibrationBtn
                    hoverEnabled: true
                    cursorShape: Qt.PointingHandCursor
                    anchors.fill: parent
                    onEntered: {
                        calibrationOverlay.opacity = 1
                    }
                    onExited: {
                        calibrationOverlay.opacity = 0
                    }
                    onClicked: {
                        dropdownHMD.enabled = true;
                        dropdownHMD.opacity = 1;
                        calibHMDitem.focus = true;
                    }
                }
                DropdownHMD {
                    id: dropdownHMD
                    x: 34
                    y: 50
                    enabled: false
                    opacity: 0
                }
            }
        }
        ColumnLayout {
            x: 121
            y:0
            width: 60
            Layout.fillHeight: false
            Layout.fillWidth: false
            Layout.alignment: Qt.AlignHCenter | Qt.AlignVCenter

            Text {
                id: calibrationStoreLabel
                text: qsTr("Store Data")
                color: "white"
                horizontalAlignment: Text.AlignHCenter
            }
            Switch {
                id: switchStore
                width: 50
                height: 40
                checked: false
                font.pointSize: 8
                onCheckedChanged: {
                    calibHMD.toggle_storage();
                }
            }
        }

        ColumnLayout {
            x: 239
            y: 0
            Layout.fillHeight: false
            Text {
                id: menuLabel
                color: "#ffffff"
                text: qsTr("Menu")
                horizontalAlignment: Text.AlignHCenter
            }

            Image {
                id: menu
                fillMode: Image.PreserveAspectFit
                z: 1
                sourceSize.width: 50
                source: "../imgs/menu.png"
                Layout.preferredHeight: 50
                sourceSize.height: 50
                Layout.preferredWidth: 50
                ColorOverlay {
                    id: menuDisabledOverlay
                    color: "#555555"
                    source: menu
                    anchors.fill: menu
                    opacity: 1
                }

                ColorOverlay {
                    id: menuOverlay
                    color: "#ffffff"
                    source: menu
                    anchors.fill: menu
                    opacity: 0
                }

                MouseArea {
                    id: menuBtn
                    hoverEnabled: true
                    anchors.fill: parent
                    cursorShape: Qt.PointingHandCursor
                    onEntered: {
                        menuOverlay.opacity = 1;
                    }
                    onExited: {
                        menuOverlay.opacity = 0;
                    }
                    onClicked: {
                        calibHMD.go_to_main_menu();
                    }
                }
            }
            Layout.alignment: Qt.AlignHCenter | Qt.AlignVCenter
            Layout.fillWidth: false
        }

    }


    /*CALIB HMD
      ----------*/
    CalibHMD {
        id: calibHMDitem
        visible: false
        width: mainWindow.width
        height: mainWindow.height
    }



    /*
    PLAYBACK CONTOL
    ------------------ */



    /*
    CAM SETTINGS
    ------------
    */
}
