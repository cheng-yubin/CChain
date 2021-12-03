// SPDX-License-Identifier: GPL-3.0
pragma solidity >=0.7.0 <0.9.0;

contract Average{

    struct control_parameters{
        address[] parties_address;      // party 地址列表
        uint iteration;                 // 正在执行的轮次
        uint gradient_size;             // 梯度规模
        uint scale;                     // float -> int 放大比例
        bool upload_or_download;        // upload: false  download: true
    }

    struct party{
        bool isPermissible;             // 权限
        bool has_uploaded;              // 本轮次是否已经上传
        bool has_download;              // 本轮次是否已经下载
        int[] num;                      // 本轮次上传的梯度数据
    }

    mapping(address => party) public parties;   // 地址到party结构的映射
    control_parameters public ctrl_params;      // 控制参数实例
    int[] public average;                       // 本轮次求得的梯度


    // 构造函数，初始化控制参数
    constructor(uint gradient_size, uint scale, address[] memory parties_address){
        ctrl_params.gradient_size = gradient_size;
        ctrl_params.iteration = 0;
        ctrl_params.scale = scale;
        ctrl_params.upload_or_download = false;     // 出于上传阶段

        for (uint i = 0; i < parties_address.length; i++){
            ctrl_params.parties_address.push(parties_address[i]);
            parties[parties_address[i]].isPermissible = true;
            parties[parties_address[i]].has_uploaded = false;
            parties[parties_address[i]].has_download = false;

            for(uint m = 0; m < ctrl_params.gradient_size; m++){
                parties[parties_address[i]].num.push(0);
            }
        }

        for(uint m = 0; m < ctrl_params.gradient_size; m++){
            average.push(0);
        }
    }


    function reset() public {
        ctrl_params.iteration = 0;
        ctrl_params.upload_or_download = false;     // 出于上传阶段
        for(uint i = 0; i < ctrl_params.parties_address.length; i++){
            parties[ctrl_params.parties_address[i]].has_uploaded = false;
            parties[ctrl_params.parties_address[i]].has_download = false;
        }

        for(uint m = 0; m < ctrl_params.gradient_size; m++){
            average[m] = 0;
        }
    }

    // 获得当前轮次
    function get_iteration() public view returns(uint){
        return ctrl_params.iteration;
    }


    // 获得合约状态：upload: false  download: true
    function get_contract_state() public view returns(bool){
        return ctrl_params.upload_or_download;
    }


    // 获得本轮次是否已经上传
    function get_has_upload() public view returns(bool){
        return parties[msg.sender].has_uploaded;
    }


    // 获得本轮次是否已经下载
    function get_has_download() public view returns(bool){
        return parties[msg.sender].has_download;
    }


    // 获得是否具有权限
    function get_has_permitted() public view returns(bool){
        return parties[msg.sender].isPermissible;
    }


    // 获得类型转换放大倍数
    function get_transform_scale() public view returns(uint){
        return ctrl_params.scale;
    }


    // 获得梯度规模
    function get_gradient_size() public view returns(uint){
        return ctrl_params.gradient_size;
    }


    function get_average_view() public view returns(int[] memory){
        return average;
    }

    // 标记已经下载梯度
    function get_average() public {
        require(parties[msg.sender].isPermissible, "No Permission");
        require(ctrl_params.upload_or_download, "not in download state");

        // 标记已经下载
        parties[msg.sender].has_download = true;

        // 判断所有party都下载了梯度
        bool flag = true;
        for(uint i = 0; i < ctrl_params.parties_address.length; i++){
            if(parties[ctrl_params.parties_address[i]].has_download == false){
                flag = false;
                break;
            }
        }
        // 当所有party都下载了梯度
        if(flag == true){
            // 进入下一轮次
            ctrl_params.iteration += 1;
            // 修改合约状态为上传模式
            ctrl_params.upload_or_download = false;
            // 清除上传和下载下载标记
            for(uint i=0; i<ctrl_params.parties_address.length; i++){
                parties[ctrl_params.parties_address[i]].has_download = false;
                parties[ctrl_params.parties_address[i]].has_uploaded = false;
            }
        }
    }


    // 上传本轮次本地梯度
    function upload_gradient(uint it, int[] memory num) public {
        require(parties[msg.sender].isPermissible, "No Permission");
        require(it == ctrl_params.iteration, "Iteration Wrong");
        require(!ctrl_params.upload_or_download, "Not in upload state");
        require(parties[msg.sender].has_uploaded == false, "You have already uploaded");
        require(num.length == ctrl_params.gradient_size, "length of gradient don't match");

        // 标记已经上传
        parties[msg.sender].has_uploaded = true;

        // 存储上传的本地梯度数据
        for(uint i=0; i<num.length; i++){
            parties[msg.sender].num[i] = num[i];
        }

        // 判断是否所有party都上传了本地梯度
        bool flag = true;
        for(uint i = 0; i < ctrl_params.parties_address.length; i++){
            if(parties[ctrl_params.parties_address[i]].has_uploaded == false){
                flag = false;
                break;
            }
        }

        // 当所有party都上传了本地梯度
        if(flag == true){
            // 计算平均值
            for(uint j=0; j<ctrl_params.gradient_size; j++){
                int sum = 0;
                for(uint i = 0; i < ctrl_params.parties_address.length; i++){
                    sum += parties[ctrl_params.parties_address[i]].num[j];
                }
                average[j] = sum / int(ctrl_params.parties_address.length);
            }
            // 修改合约状态为下载模式
            ctrl_params.upload_or_download = true;
        }
    }
}