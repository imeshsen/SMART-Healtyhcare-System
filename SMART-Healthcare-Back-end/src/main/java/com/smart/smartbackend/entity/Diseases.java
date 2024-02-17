package com.smart.smartbackend.entity;

import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;

@AllArgsConstructor
@NoArgsConstructor
@Data

public class Diseases {
    private String type;
    private String name;
    private String specialist;
}
