package com.smart.smartbackend.entity;

import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;

@AllArgsConstructor
@NoArgsConstructor
@Data

public class Register {
    private String name;
    private String nic;
    private int mobile;
    private String hometown;
}
