// Export local analysis artifacts; do not commit vendor-derived decompilation.
// @category NX Ctudio

import ghidra.app.decompiler.DecompInterface;
import ghidra.app.decompiler.DecompileResults;
import ghidra.app.script.GhidraScript;
import ghidra.program.model.address.Address;
import ghidra.program.model.listing.Function;
import ghidra.program.model.listing.FunctionIterator;
import ghidra.program.model.symbol.Reference;
import java.io.BufferedWriter;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;

public class ExportFunctions extends GhidraScript {
    @Override
    public void run() throws Exception {
        String[] args = getScriptArgs();
        if (args.length != 3) {
            throw new IllegalArgumentException("Expected output directory, start VA, end VA.");
        }
        Path output = Path.of(args[0]);
        Files.createDirectories(output);
        Address start = toAddr(args[1]);
        Address end = toAddr(args[2]);
        DecompInterface decompiler = new DecompInterface();
        try {
            if (!decompiler.openProgram(currentProgram)) {
                throw new IllegalStateException(decompiler.getLastMessage());
            }
            try (BufferedWriter index = Files.newBufferedWriter(
                    output.resolve("functions.tsv"), StandardCharsets.UTF_8)) {
                index.write("address\tname\tstatus\n");
                FunctionIterator functions = currentProgram.getFunctionManager().getFunctions(true);
                while (functions.hasNext()) {
                    monitor.checkCancelled();
                    Function function = functions.next();
                    Address address = function.getEntryPoint();
                    if (address.compareTo(start) < 0 || address.compareTo(end) >= 0) {
                        continue;
                    }
                    DecompileResults result = decompiler.decompileFunction(function, 60, monitor);
                    String status = result.decompileCompleted() ? "ok" : result.getErrorMessage();
                    index.write(address + "\t" + function.getName() + "\t" + status + "\n");
                    index.flush();
                    if (result.decompileCompleted()) {
                        Files.writeString(output.resolve(address + ".c"),
                                result.getDecompiledFunction().getC(), StandardCharsets.UTF_8);
                    }
                    try (BufferedWriter refs = Files.newBufferedWriter(
                            output.resolve(address + ".refs.txt"), StandardCharsets.UTF_8)) {
                        for (Reference reference : getReferencesTo(address)) {
                            refs.write(reference.toString() + "\n");
                        }
                    }
                }
            }
        } finally {
            decompiler.dispose();
        }
    }
}
