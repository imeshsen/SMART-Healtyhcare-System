package com.smart.smartbackend.controller;

import com.smart.smartbackend.entity.Register;
import com.smart.smartbackend.service.PatientService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.web.bind.annotation.*;

import java.util.List;
import java.util.concurrent.ExecutionException;

@RestController
@RequestMapping(value = "api/v1/patient")
@CrossOrigin
public class Controller {
    @Autowired
    public PatientService patientService;

    @GetMapping("/print")
    public String printuser() {
        return "imesh";
    }

    @PostMapping("/save")
    public String savePatient(@RequestBody Register register) throws ExecutionException, InterruptedException {
        return patientService.savePatient(register);
    }

    @GetMapping("/search/{nic}")
    public Register searchPatient(@PathVariable String nic) throws ExecutionException, InterruptedException {
        return patientService.searchPatient(nic);
    }

    @GetMapping("/patients/")
    public List<Register> getAllPatients() throws ExecutionException, InterruptedException {
        return patientService.getAllPatients();
    }

}
