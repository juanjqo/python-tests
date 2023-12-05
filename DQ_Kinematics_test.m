clear all;
close all;
include_namespace_dq

NUMBER_OF_RANDOM = 1000;

serial_manipulator = KukaLwr4Robot.kinematics();
whole_body = KukaYoubot.kinematics();

%% Create a serial-whole-body robot
serial_whole_body = DQ_SerialWholeBody(serial_manipulator);
serial_whole_body.set_reference_frame(1 + 0.5*E_*(0.5*k_));
serial_whole_body.add(FrankaEmikaPandaRobot.kinematics());

%% Generate data for unary and binary operators
random_q = random('unif',-pi,pi,[7 NUMBER_OF_RANDOM]);
random_q_dot = random('unif',-pi,pi,[7 NUMBER_OF_RANDOM]);
random_q_whole_body = random('unif',-pi,pi,[8 NUMBER_OF_RANDOM]);
random_q_serial_whole_body = random('unif',-pi,pi,[serial_whole_body.get_dim_configuration_space() NUMBER_OF_RANDOM]);
random_dq_a = random('unif',-10,10,[8 NUMBER_OF_RANDOM]);
%random_dq_b = random('unif',-10,10,[8 NUMBER_OF_RANDOM]);

%% Pre-allocate results
result_of_fkm = zeros(8, NUMBER_OF_RANDOM);
result_of_pose_jacobian = zeros(8, 7, NUMBER_OF_RANDOM);
result_of_pose_jacobian_derivative = zeros(8, 7, NUMBER_OF_RANDOM);

result_of_whole_body_fkm = zeros(8, NUMBER_OF_RANDOM);
result_of_whole_body_jacobian = zeros(8, 8, NUMBER_OF_RANDOM);

result_of_serial_whole_body_fkm      = zeros(8, NUMBER_OF_RANDOM);
result_of_serial_whole_body_raw_fkm  = zeros(8, NUMBER_OF_RANDOM);
result_of_serial_whole_body_jacobian = zeros(8, serial_whole_body.get_dim_configuration_space(), NUMBER_OF_RANDOM);

%       distance_jacobian - Compute the (squared) distance Jacobian.
result_of_distance_jacobian = zeros(1, 7, NUMBER_OF_RANDOM);
%       rotation_jacobian - Compute the rotation Jacobian.
result_of_rotation_jacobian = zeros(4, 7, NUMBER_OF_RANDOM);
%       translation_jacobian - Compute the translation Jacobian.
result_of_translation_jacobian = zeros(4, 7, NUMBER_OF_RANDOM);
%       line_jacobian - Compute the line Jacobian.
result_of_line_jacobian = zeros(8, 7, NUMBER_OF_RANDOM);
%       plane_jacobian - Compute the plane Jacobian.
result_of_plane_jacobian = zeros(8, 7, NUMBER_OF_RANDOM);


%       line_to_line_distance_jacobian  - Compute the line-to-line distance Jacobian.
result_of_line_to_line_distance_jacobian = zeros(1, 7, NUMBER_OF_RANDOM);
%       line_to_line_residual - Compute the line-to-line residual.

%       line_to_point_distance_jacobian - Compute the line-to-line distance Jacobian.
result_of_line_to_point_distance_jacobian = zeros(1, 7, NUMBER_OF_RANDOM);
%       line_to_point_residual - Compute the line-to-point residual.

%       plane_to_point_distance_jacobian - Compute the plane-to-point distance Jacobian.
result_of_plane_to_point_distance_jacobian = zeros(1, 7, NUMBER_OF_RANDOM);
%       plane_to_point_residual - Compute the plane-to-point residual.

%       point_to_line_distance_jacobian - Compute the point-to-line distance Jacobian.
result_of_point_to_line_distance_jacobian = zeros(1, 7, NUMBER_OF_RANDOM);
%       point_to_line_residual - Compute the point to line residual.

%       point_to_plane_distance_jacobian - Compute the point to plane distance Jacobian.
result_of_point_to_plane_distance_jacobian = zeros(1, 7, NUMBER_OF_RANDOM);
%       point_to_plane_residual - Compute the point to plane residual.


%       point_to_point_distance_jacobian - Compute the point to point distance Jacobian.
result_of_point_to_point_distance_jacobian = zeros(1, 7, NUMBER_OF_RANDOM);
%       point_to_point_residual - Compute the point to point residual.

%       line_to_line_angle_jacobian - Compute the line-to-line angle Jacobian.
result_of_line_to_line_angle_jacobian = zeros(1, 7, NUMBER_OF_RANDOM);



%% Loop
for i=1:NUMBER_OF_RANDOM
    %% Preliminaries
    ha = DQ(random_dq_a(:,i));
    
    %% DQ_SerialManipulator
    x_pose = serial_manipulator.fkm(random_q(:,i));
    J_pose = serial_manipulator.pose_jacobian(random_q(:,i));
    
    %% Preliminaries for DQ_Kinematics
    robot_point = translation(x_pose);
    robot_line = get_line_from_dq(x_pose, k_);
    robot_plane = get_plane_from_dq(x_pose, k_);
    workspace_line = get_line_from_dq(ha, k_);
    workspace_point = translation(normalize(ha));
    workspace_plane = get_plane_from_dq(ha, k_);
    translation_jacobian = DQ_Kinematics.translation_jacobian(J_pose, x_pose);
    line_jacobian = DQ_Kinematics.line_jacobian(J_pose, x_pose, k_);
    plane_jacobian = DQ_Kinematics.plane_jacobian(J_pose, x_pose, k_);
    
    %% Results of DQ_SerialManipulator
    result_of_fkm(:, i) = vec8(x_pose);
    result_of_pose_jacobian(:,:,i) = J_pose;
    result_of_pose_jacobian_derivative(:,:,i) = serial_manipulator.pose_jacobian_derivative(random_q(:,i),random_q_dot(:,i));
    %% Results of DQ_SerialWholeBody
    result_of_serial_whole_body_fkm(:, i)       = vec8(serial_whole_body.fkm(random_q_serial_whole_body(:,i)));
    result_of_serial_whole_body_raw_fkm(:, i)   = vec8(serial_whole_body.raw_fkm(random_q_serial_whole_body(:,i)));
    result_of_serial_whole_body_jacobian(:,:,i) = serial_whole_body.pose_jacobian(random_q_serial_whole_body(:,i));
    %% Results of DQ_WholeBody
    result_of_whole_body_fkm(:, i) = vec8(whole_body.fkm(random_q_whole_body(:,i)));
    result_of_whole_body_jacobian(:,:,i) = whole_body.pose_jacobian(random_q_whole_body(:,i));
    %% Results of DQ_Kinematics
    result_of_distance_jacobian(:,:,i) = DQ_Kinematics.distance_jacobian(J_pose,x_pose);
    result_of_rotation_jacobian(:,:,i) = DQ_Kinematics.rotation_jacobian(J_pose);
    result_of_translation_jacobian(:,:,i) = translation_jacobian;
    result_of_line_jacobian(:,:,i) = line_jacobian;
    result_of_plane_jacobian(:,:,i) = plane_jacobian;
    result_of_line_to_line_distance_jacobian(:,:,i) = DQ_Kinematics.line_to_line_distance_jacobian(line_jacobian,robot_line,workspace_line);
    result_of_line_to_point_distance_jacobian(:,:,i) = DQ_Kinematics.line_to_point_distance_jacobian(line_jacobian,robot_line,workspace_point);
    result_of_plane_to_point_distance_jacobian(:,:,i) = DQ_Kinematics.plane_to_point_distance_jacobian(plane_jacobian,workspace_point);
    result_of_point_to_line_distance_jacobian(:,:,i) = DQ_Kinematics.point_to_line_distance_jacobian(translation_jacobian,robot_point,workspace_line);
    result_of_point_to_plane_distance_jacobian(:,:,i) = DQ_Kinematics.point_to_plane_distance_jacobian(translation_jacobian,robot_point,workspace_plane);
    result_of_point_to_point_distance_jacobian(:,:,i) = DQ_Kinematics.point_to_point_distance_jacobian(translation_jacobian,robot_point,workspace_point);
    result_of_line_to_line_angle_jacobian(:,:,i) = DQ_Kinematics.line_to_line_angle_jacobian(line_jacobian,robot_line,workspace_line);
end

save DQ_Kinematics_test.mat

function ret = get_line_from_dq(dq, primitive)
include_namespace_dq
dq = normalize(dq);
ret = Ad(rotation(dq), primitive) + E_*cross(translation(dq), Ad(rotation(dq), primitive));
end

function ret = get_plane_from_dq(dq, primitive)
include_namespace_dq
dq = normalize(dq);
ret = Ad(rotation(dq), primitive) + E_*dot(translation(dq), Ad(rotation(dq), primitive));
end